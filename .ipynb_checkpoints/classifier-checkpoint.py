from typing import List, Dict
import json
import torch
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches

class SmartQueryClassifier:
    def __init__(self, llm_model, llm_tokenizer):
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.categories = {
            "performance": {
                "keywords": [
                    "performance", "flops", "flop/s", "operations", "throughput", 
                    "compute", "gflops", "tflops", "peak performance", "speed",
                    "instructions per cycle", "ipc", "clock", "frequency",
                    "fp64", "fp32", "avx", 
                    "double-precision performance", "single-precision performance"
                ],
                "description": "Questions about computational performance metrics"
            },
            "memory": {
                "keywords": [
                    "memory bandwidth", "bandwidth", "gb/s", "memory speed", 
                    "dram", "hbm", "cache", "memory latency", "memory throughput",
                    "memory performance", "memory access"
                ],
                "description": "Questions about memory bandwidth"
            }
        }

    def classify_with_llm(self, question: str) -> Dict[str, float]:
        prompt = f"""Analyze the following processor-related question and determine which technical aspects it primarily addresses. 
Consider the context of processor specifications and performance characteristics.

For each of these categories, provide a confidence score from 0.0 to 1.0:
- performance: Computational performance metrics (FLOPs, throughput, IPC)
- memory: Memory characteristics (bandwidth, latency, cache)
- other: Any question that does not fall into the above categories

Return ONLY a JSON dictionary with the categories as keys and confidence scores as values.

Question: {question}

Analysis:"""

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("🔎 Raw LLM Output:\n", response)

        try:
            import re
            json_str_match = re.search(r'{.*}', response, re.DOTALL)
            if json_str_match:
                json_str = json_str_match.group().replace("'", '"')
                return json.loads(json_str)
            else:
                raise json.JSONDecodeError("No valid JSON found", response, 0)
        except json.JSONDecodeError:
            # print("⚠️ LLM parsing failed. Falling back to keyword + semantic + fuzzy classification.")
            return self.classify_with_all_fallbacks(question)

    def classify_with_keywords(self, question: str) -> Dict[str, float]:
        scores = {category: 0.0 for category in self.categories}
        question_lower = question.lower()
        
        for category, data in self.categories.items():
            for keyword in data["keywords"]:
                if keyword in question_lower:
                    scores[category] += 0.2
        
        max_score = max(scores.values())
        if max_score > 0:
            for category in scores:
                scores[category] = scores[category] / max_score
        return scores

    def classify_with_semantic_similarity(self, question: str, threshold: float = 0.6) -> Dict[str, float]:
        scores = {category: 0.0 for category in self.categories}
        question_embedding = self.embedder.encode(question, convert_to_tensor=True)

        for category, data in self.categories.items():
            keyword_embeddings = self.embedder.encode(data["keywords"], convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(question_embedding, keyword_embeddings)
            max_score = float(torch.max(cosine_scores))
            if max_score > threshold:
                scores[category] = max_score
        return scores

    def classify_with_fuzzy_matching(self, question: str, threshold: float = 0.7) -> Dict[str, float]:
        scores = {category: 0.0 for category in self.categories}
        question_words = question.lower().split()

        for category, data in self.categories.items():
            for keyword in data["keywords"]:
                matches = get_close_matches(keyword, question_words, n=1, cutoff=threshold)
                if matches:
                    scores[category] += 0.2
        max_score = max(scores.values())
        if max_score > 0:
            for category in scores:
                scores[category] = scores[category] / max_score
        return scores

    def classify_with_all_fallbacks(self, question: str) -> Dict[str, float]:
        kw_scores = self.classify_with_keywords(question)
        sem_scores = self.classify_with_semantic_similarity(question)
        fuzzy_scores = self.classify_with_fuzzy_matching(question)

        combined = {}
        for category in self.categories:
            combined[category] = max(kw_scores.get(category, 0), sem_scores.get(category, 0), fuzzy_scores.get(category, 0))
        return combined

    def get_top_categories(self, question: str, threshold: float = 0.5) -> List[str]:
        scores = self.classify_with_llm(question)
        return [category for category, score in scores.items() if score >= threshold]

    def is_performance_question(self, question: str) -> bool:
        scores = self.classify_with_llm(question)
        return scores.get("performance", 0.0) >= 0.5

    def is_memory_question(self, question: str) -> bool:
        scores = self.classify_with_llm(question)
        return scores.get("memory", 0.0) >= 0.5
