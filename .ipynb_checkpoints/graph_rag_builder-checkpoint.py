import os
import json
import re
import networkx as nx
from rapidfuzz import fuzz, process
from rapidfuzz.fuzz import token_sort_ratio
from pathlib import Path

# Ensure absolute path regardless of where the script is run
base_path = Path(__file__).resolve().parent
json_folder_path = base_path / "data2"
G = nx.DiGraph()
KNOWN_MODEL_IDS = set()

def parse_description_attributes(text):
    fields = {}
    if "server/workstation" in text:
        fields["usage_type"] = "server/workstation"
    if m := re.search(r"Socket\s+([A-Z0-9']+)", text):
        fields["socket_type"] = m.group(1)
    if m := re.search(r"part of the (.+?) lineup", text, re.IGNORECASE):
        fields["lineup"] = m.group(1).strip()
    if "extremely power hungry" in text:
        fields["power_profile"] = "extremely_power_hungry"
    elif "power hungry" in text:
        fields["power_profile"] = "power_hungry"
    if "Hyper-Threading" in text or "Simultaneous Multithreading" in text:
        fields["threading_tech"] = "SMT/HT"
    if "lacks integrated graphics" in text:
        fields["integrated_graphics"] = False
    if "multi-processor" in text or "multi-socket" in text:
        fields["multi_socket_support"] = True
    if m := re.search(r"(\d+(?:\.\d+)?)\s*MB of L3 cache", text):
        fields["L3_cache_mb"] = float(m.group(1))
    if m := re.search(r"on a (\d+)\s*nm production", text):
        fields["process_node_nm"] = int(m.group(1))
    if "Advanced Vector Extensions (AVX)" in text:
        fields["avx_supported"] = True
    for instr in ["AVX2", "AVX-512"]:
        if instr in text:
            fields[f"{instr.lower()}_supported"] = True
    return fields

def add_json_to_graph(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    for data in json_list:
        model_id = data["model_id"]
        KNOWN_MODEL_IDS.add(model_id)
        G.add_node(model_id, type="processor")

        for key, val in data.get("specs", {}).items():
            if val is not None:
                node = f"{key}:{val}"
                G.add_node(node, type="attribute", attribute_key=key, value=val)
                G.add_edge(model_id, node, relation="has_attribute")

        for feature in data.get("features", []):
            node = f"feature:{feature}"
            G.add_node(node, type="feature", value=feature)
            G.add_edge(model_id, node, relation="has_feature")

        desc_text = data.get("description", "")
        if desc_text.strip():
            desc_node = f"description:{model_id}"
            G.add_node(desc_node, type="description", content=desc_text)
            G.add_edge(model_id, desc_node, relation="has_description")

            for k, v in parse_description_attributes(desc_text).items():
                if v is not None:
                    node = f"{k}:{v}"
                    G.add_node(node, type="parsed_attribute", attribute_key=k, value=v)
                    G.add_edge(model_id, node, relation="has_parsed_attribute")

        for field in ["brand", "codename", "generation", "release_date", "price_usd", "socket_number"]:
            val = data.get(field)
            if val:
                node = f"{field}:{val}"
                G.add_node(node, type="metadata", attribute_key=field, value=val)
                G.add_edge(model_id, node, relation="has_metadata")

        try:
            tdp = float(data["specs"].get("tdp_watts", 0))
            gflops = float(data["specs"].get("peak_performance_gflops", 0))
            if tdp > 0 and gflops > 0:
                eff = round(gflops / tdp, 2)
                node = f"performance_per_watt:{eff}"
                G.add_node(node, type="computed", value=eff)
                G.add_edge(model_id, node, relation="has_computed_attribute")
        except Exception:
            pass

def load_all_jsons():
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            add_json_to_graph(os.path.join(json_folder_path, filename))

def get_subgraph_for_model(model_id, radius=2):
    if model_id not in G:
        return None
    return nx.ego_graph(G, model_id, radius=radius, undirected=False)

def subgraph_to_prompt(subgraph, terms: list[str] = None):
    grouped = {
        "Compute": [],
        "Memory": [],
        "I/O": [],
        "Vector Extensions": [],
        "Description": "",
        "Architecture & Meta": [],
        "Other": []
    }

    norm_terms = [t.lower() for t in terms] if terms else None

    for u, v, d in subgraph.edges(data=True):
        relation = d.get("relation")
        if ":" not in v:
            continue

        key, val = v.split(":", 1)
        text_to_match = f"{key}:{val}".lower()

        # 🧠 FILTERING: if terms are provided, check overlap
        if norm_terms and not any(t in text_to_match for t in norm_terms):
            continue

        if relation == "has_attribute":
            if "core" in v or "thread" in v or "clock" in v or "fp_unit" in v or "simd" in v:
                grouped["Compute"].append(f"{u} {relation} {v}")
            elif "memory" in v or "hbm" in v:
                grouped["Memory"].append(f"{u} {relation} {v}")
            elif "pcie" in v:
                grouped["I/O"].append(f"{u} {relation} {v}")
            else:
                grouped["Other"].append(f"{u} {relation} {v}")
        elif relation == "has_feature":
            if "avx" in v.lower():
                grouped["Vector Extensions"].append(f"{u} {relation} {v}")
            else:
                grouped["Other"].append(f"{u} {relation} {v}")
        elif relation in {"has_parsed_attribute", "has_metadata"}:
            grouped["Architecture & Meta"].append(f"{u} {relation} {v}")
        elif relation == "has_computed_attribute":
            grouped["Other"].append(f"{u} {relation} {v}")
        elif relation == "has_description":
            desc = subgraph.nodes[v].get("content", "").strip()
            if desc and (not norm_terms or any(t in desc.lower() for t in norm_terms)):
                grouped["Description"] = f"\n🔎 Description of {u}:\n{desc}"

    prompt_sections = []
    for title, lines in grouped.items():
        if lines:
            if isinstance(lines, str):
                prompt_sections.append(lines)
            else:
                prompt_sections.append(f"\n🔹 {title}:\n" + "\n".join(lines))
    return "\n".join(prompt_sections)


def get_comparison_context(model_ids: list[str], radius: int = 2) -> str:
    sections = []

    for model_id in model_ids:
        subgraph = get_subgraph_for_model(model_id, radius)
        if not subgraph:
            continue

        grouped = {
            "Compute": [],
            "Memory": [],
            "I/O": [],
            "Vector Extensions": [],
            "Architecture & Meta": [],
            "Other": [],
            "Description": ""
        }

        for u, v, d in subgraph.edges(data=True):
            relation = d.get("relation")
            if ":" not in v:
                continue

            if relation == "has_attribute":
                if any(kw in v for kw in ["core", "thread", "clock", "fp_unit", "simd"]):
                    grouped["Compute"].append(f"{model_id} {relation} {v}")
                elif "memory" in v or "hbm" in v:
                    grouped["Memory"].append(f"{model_id} {relation} {v}")
                elif "pcie" in v:
                    grouped["I/O"].append(f"{model_id} {relation} {v}")
                else:
                    grouped["Other"].append(f"{model_id} {relation} {v}")
            elif relation == "has_feature":
                grouped["Vector Extensions"].append(f"{model_id} {relation} {v}")
            elif relation in ["has_metadata", "has_parsed_attribute", "has_computed_attribute"]:
                grouped["Architecture & Meta"].append(f"{model_id} {relation} {v}")
            elif relation == "has_description":
                desc = subgraph.nodes[v].get("content", "").strip()
                # Only assign description once
                if desc and not grouped["Description"]:
                    grouped["Description"] = f"\n🔎 Description of {model_id}:\n{desc}"

        # Format for this model
        section = [f"\nModel: {model_id}"]
        for title, lines in grouped.items():
            if lines:
                if isinstance(lines, str):
                    section.append(lines)
                else:
                    section.append(f"\n🔹 {title}:\n" + "\n".join(lines))
        sections.append("\n".join(section))

    return "\n\n".join(sections)

def build_sectionwise_chunks_from_graph():
    chunks = []
    for model_id in G.nodes:
        if G.nodes[model_id].get("type") != "processor":
            continue

        subgraph = get_subgraph_for_model(model_id)
        if not subgraph:
            continue

        sections = {
            "Compute": [],
            "Memory": [],
            "I/O": [],
            "Vector Extensions": [],
            "Architecture & Meta": [],
            "Other": [],
            "Description": ""
        }

        for u, v, d in subgraph.edges(data=True):
            relation = d.get("relation", "")
            if ":" not in v:
                continue

            if relation == "has_attribute":
                if any(k in v for k in ["core", "thread", "clock", "fp_unit", "simd"]):
                    sections["Compute"].append(f"{model_id} {relation} {v}")
                elif "memory" in v or "hbm" in v:
                    sections["Memory"].append(f"{model_id} {relation} {v}")
                elif "pcie" in v:
                    sections["I/O"].append(f"{model_id} {relation} {v}")
                else:
                    sections["Other"].append(f"{model_id} {relation} {v}")
            elif relation == "has_feature":
                sections["Vector Extensions"].append(f"{model_id} {relation} {v}")
            elif relation in ["has_metadata", "has_parsed_attribute", "has_computed_attribute"]:
                sections["Architecture & Meta"].append(f"{model_id} {relation} {v}")
            elif relation == "has_description":
                desc = subgraph.nodes[v].get("content", "").strip()
                if desc and desc not in sections["Description"]:
                    sections["Description"] = f"\n🔎 Description of {model_id}:\n{desc}\n"


        # Convert each section into a separate chunk
        for section, lines in sections.items():
            if not lines:
                continue
            chunk_text = lines if isinstance(lines, str) else "\n".join(lines)
            chunk_id = f"{model_id}_{section.replace(' ', '_')}"
            chunks.append({"model_id": model_id, "section": section, "text": chunk_text})

    return chunks


def get_graph_prompt(model_id: str, radius: int = 2, terms: list[str] = None) -> str:
    if not G.nodes:
        load_all_jsons()
    subgraph = get_subgraph_for_model(model_id, radius)
    return subgraph_to_prompt(subgraph, terms) if subgraph else f"⚠️ Model ID {model_id} not found in the graph."   
#add the logic for smart model_ids extractor

# 🧠 MAIN: Ask a question, extract model ID, get context, print result
if __name__ == "__main__":
    load_all_jsons()

    question = input("Your question: ").strip()
    model_id = extract_model_id_from_question(question)

    if not model_id:
        print("❌ Could not detect a valid model ID in your question.")
    else:
        context = get_graph_prompt(model_id)
        # print(f"\n📦 Auto-detected Model ID: {model_id}")
        # print(f"\n🔎 Context used for answering:\n{'='*60}")
        # print(context)
