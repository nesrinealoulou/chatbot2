from graph.graph_rag_builder import get_subgraph_for_model

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
                if desc and desc not in grouped["Description"]:
                    grouped["Description"] = f"\n🔎 Description of {model_id}:\n{desc}"

        section = [f"\nModel: {model_id}"]
        for title, lines in grouped.items():
            if lines:
                section.append(f"\n🔹 {title}:\n" + "\n".join(lines) if not isinstance(lines, str) else lines)
        sections.append("\n".join(section))

    return "\n\n".join(sections)
