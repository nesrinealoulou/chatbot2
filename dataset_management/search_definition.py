import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

# -------------------
# 📁 Path Definitions
# -------------------
embedding_path = "data3/definition_embeddings.npy"
index_path = "data3/definition_index.json"
text_path = "data3/definition_texts.json"
definition_json_path = "data3/definitions.json"
term_embeddings_path = "data3/term_embeddings.npy"

# -------------------
# 🔍 Load LLM Embedding Model
# -------------------
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("✅ Model loaded.")

# -------------------
# 🧠 Create Embeddings (if not already saved)
# -------------------
if not os.path.exists(embedding_path) or not os.path.exists(index_path) or not os.path.exists(text_path):
    print("📦 Embedding files not found — generating from definitions.json...")

    with open(definition_json_path, "r", encoding="utf-8") as f:
        definitions = json.load(f)

    texts = [f"{item['term']}: {item['definition']}" for item in definitions]
    embeddings = model.encode(texts, normalize_embeddings=True)

    np.save(embedding_path, embeddings)
    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)

    print("✅ Embeddings and metadata saved.")

# Generate term embeddings if not exists
if not os.path.exists(term_embeddings_path):
    with open(definition_json_path, "r", encoding="utf-8") as f:
        definitions = json.load(f)
    terms = [item["term"] for item in definitions]
    term_embeddings = model.encode(terms, normalize_embeddings=True)
    np.save(term_embeddings_path, term_embeddings)
    print("✅ Term embeddings saved.")

# -------------------
# 🧠 Load Data
# -------------------
embeddings = np.load(embedding_path)
with open(index_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

texts = [f"{item['term']}: {item['definition']}" for item in metadata]
terms = [item["term"] for item in metadata]
term_embeddings = np.load(term_embeddings_path)

# -------------------
# ⚙️ Build FAISS Indexes
# -------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

term_dimension = term_embeddings.shape[1]
term_index = faiss.IndexFlatIP(term_dimension)
term_index.add(term_embeddings)

# -------------------
# 🔍 Extract Terms from Query Using Embeddings Only
# -------------------
def extract_terms_from_query(query, top_k=5, threshold=0.6):
    query_embedding = model.encode(query, normalize_embeddings=True).reshape(1, -1)
    D, I = term_index.search(query_embedding, top_k * 3)

    extracted = []
    for score, idx in zip(D[0], I[0]):
        if score >= threshold:
            extracted.append((terms[idx], float(score)))

    return [term for term, _ in sorted(extracted, key=lambda x: x[1], reverse=True)[:top_k]]

# -------------------
# 🔍 Main Definition Search Function
# -------------------
def search_definitions(query, top_k=3):
    extracted_terms = extract_terms_from_query(query)
    if not extracted_terms:
        return {"overall_results": [], "term_results": {}}

    term_results = {}
    for term in extracted_terms:
        term_embedding = model.encode(term, normalize_embeddings=True).reshape(1, -1)
        D, I = index.search(term_embedding, top_k)

        term_results[term] = [
            {
                "term": metadata[idx]["term"],
                "definition": metadata[idx]["definition"],
                "score": float(score)
            }
            for score, idx in zip(D[0], I[0])
        ]

    query_embedding = model.encode(query, normalize_embeddings=True).reshape(1, -1)
    D, I = index.search(query_embedding, top_k)
    overall_results = [
        {
            "term": metadata[idx]["term"],
            "definition": metadata[idx]["definition"],
            "score": float(score)
        }
        for score, idx in zip(D[0], I[0])
    ]

    return {"overall_results": overall_results, "term_results": term_results}

# -------------------
# 🔁 Interactive CLI
# -------------------
if __name__ == "__main__":
    print("💬 Ask about any CPU/GPU concept (type 'exit' to quit):")
    while True:
        query = input("\n🧠 Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        results = search_definitions(query)

        if not results["overall_results"] and not results["term_results"]:
            print("❌ No relevant definitions found.")
            continue

        if results["term_results"]:
            print("\n🔍 Detected Terms and Their Definitions:")
            for term, matches in results["term_results"].items():
                print(f"\n📌 Term: {term}")
                for i, match in enumerate(matches[:3], 1):
                    print(f"{i}. {match['term']}: {match['definition']}")

        if results["overall_results"]:
            print("\n📘 Overall Best Matches:")
            for i, item in enumerate(results["overall_results"][:3], 1):
                print(f"{i}. {item['term']}: {item['definition']}")
