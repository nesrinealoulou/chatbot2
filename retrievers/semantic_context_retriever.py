import numpy as np
import faiss

def build_line_index(chunks, embed_model):
    line_embeddings, line_texts, line_sources = [], [], []

    for chunk in chunks:
        model_id = chunk["model_id"]
        section = chunk["section"]
        for line in chunk["text"].splitlines():
            line = line.strip()
            if not line:
                continue
            line_texts.append(line)
            line_sources.append((model_id, section))
            line_embeddings.append(embed_model.encode(line))

    index = faiss.IndexFlatL2(len(line_embeddings[0]))
    index.add(np.array(line_embeddings))

    return index, line_texts, line_sources

def search_semantic_lines(term_list, embed_model, index, line_texts, line_sources, model_ids=None, top_k=5):
    matched_lines = set()
    for term in term_list:
        query_emb = embed_model.encode(term)
        D, I = index.search(np.array([query_emb]), top_k)
        for idx in I[0]:
            model_id, _ = line_sources[idx]
            if model_ids is None or model_id in model_ids:
                matched_lines.add(line_texts[idx])
    return list(matched_lines)
