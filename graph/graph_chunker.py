from graph.graph_rag_builder import get_subgraph_for_model, G

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