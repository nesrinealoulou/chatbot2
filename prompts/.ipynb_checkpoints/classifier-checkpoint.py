# intelligent_classifier.py

from typing import Dict, List
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class IntelligentQueryClassifier:
    def __init__(self):
        # Embedding model for semantic similarity
        self.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.threshold = 0.7

        # LLM clarifier & classifier
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")

        # Coarse categories descriptions
        self.coarse_categories = {
            "performance": "Questions about computational performance like GFLOPS, IPC, clock speeds.",
            "memory": "Questions about memory bandwidth, latency, throughput, caches.",
            "general": "General questions about processors or others that do not fit performance or memory."
        }

        # Precompute embeddings for coarse categories
        self.coarse_category_embeddings = {
            cat: self.embedder.encode(desc, convert_to_tensor=True)
            for cat, desc in self.coarse_categories.items()
        }

    def clarify_question(self, question: str) -> str:
        prompt = f"Fix typos and clarify the user query to make it understandable: {question}"
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=64)
        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def llm_vote_coarse(self, question: str) -> str:
        prompt = f"""
You are an expert in processor architecture.

Classify the user question into one of the following coarse categories:
- performance
- memory
- general

User Question: "{question}"

Respond with only the category name.
"""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=8)
        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    def classify_coarse(self, question: str) -> Dict[str, float]:
        clarified = self.clarify_question(question)
        print(f"🔍 Clarified Question: {clarified}")
    
        q_embed = self.embedder.encode(clarified, convert_to_tensor=True)
        scores = {}
        for cat, emb in self.coarse_category_embeddings.items():
            sim = util.cos_sim(q_embed, emb).item()
            scores[cat] = round(sim, 3)
    
        llm_prediction = self.llm_vote_coarse(clarified)
        if llm_prediction in scores:
            scores[llm_prediction] += 0.2  # Boost LLM-voted category
            scores[llm_prediction] = min(scores[llm_prediction], 1.0)
    
        # 🛠️ Enriched keyword override logic
        lower_q = clarified.lower()
    
        performance_keywords = [
            "fp64", "fp32", "performance", "gflops", "tflops", "compute", "operations per second",
            "instructions", "execution", "vectorization", "scalar",
            "floating point", "core speed", "processing power", "per-core fp", "per-core performance","high-throughput computation capacity"
        ]
    
        memory_keywords = [
            "memory", "bandwidth", "latency", "cache", "ram", "dram", "hbm", "ddr",
            "throughput", "memory access", "bus speed", "numa", "load/store"
        ]
    
        matched_categories = {}
    
        if any(k in lower_q for k in performance_keywords):
            print("⚙️ Keyword match: adding category 'performance'")
            matched_categories["performance"] = 1.0
    
        if any(k in lower_q for k in memory_keywords):
            print("⚙️ Keyword match: adding category 'memory'")
            matched_categories["memory"] = 1.0
    
        if matched_categories:
            return matched_categories
    
        if scores:
            return scores
    
        print("⚠️ No relevant category found. Defaulting to 'general'")
        return {"general": 1.0}



    def get_top_coarse_categories(self, question: str) -> List[str]:
        scores = self.classify_coarse(question)
        print(f"🔍 Coarse Category Scores for: '{question}'\n{scores}")
        
        # Always return categories with score above threshold, or fallback to general
        top_cats = [cat for cat, score in scores.items() if score >= self.threshold]
        return top_cats if top_cats else ["general"]

