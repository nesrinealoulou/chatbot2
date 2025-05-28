import numpy as np
import faiss

chunks = []
chunk_texts = []
chunk_embeddings = None
index = None
embed_model = None

def initialize_embedding_index(model, precomputed_chunks):
    global chunks, chunk_texts, chunk_embeddings, index, embed_model
    chunks = precomputed_chunks
    embed_model = model
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = embed_model.encode(chunk_texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
    index.add(np.array(chunk_embeddings))

def retrieve_chunks_for_model_ids(model_ids: list[str], question: str, top_k=3):
    model_chunks = [c for c in chunks if c["model_id"] in model_ids]
    texts = [c["text"] for c in model_chunks]
    embeddings = embed_model.encode(texts, convert_to_tensor=False)
    local_index = faiss.IndexFlatL2(len(embeddings[0]))
    local_index.add(np.array(embeddings))
    q_emb = embed_model.encode(question, convert_to_tensor=False)
    D, I = local_index.search(np.array([q_emb]), top_k)
    return [texts[i] for i in I[0]]

def retrieve_relevant_context(question, top_k=3, verbose=True):
    q_embedding = embed_model.encode(question, convert_to_tensor=False)
    D, I = index.search(np.array([q_embedding]), top_k)
    if verbose:
        return [f"🔹 {chunks[i]['section']} of {chunks[i]['model_id']}:\n{chunks[i]['text']}" for i in I[0]]
    else:
        return [chunks[i]["text"] for i in I[0]]

def compute_known_model_embeddings(known_model_ids):
    model_id_list = list(known_model_ids)
    embeddings = embed_model.encode(model_id_list, convert_to_tensor=False)
    return {model_id: emb for model_id, emb in zip(model_id_list, embeddings)}
