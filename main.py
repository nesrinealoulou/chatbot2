import torch
from typing import List
from model_manager import model_manager
from extractors.extract_proc import extract_models_only
from graph.graph_comparator import get_comparison_context
from graph.graph_rag_builder import get_graph_prompt
from models.embedding_utils import retrieve_chunks_for_model_ids, retrieve_relevant_context
from extractors.formula_attr_extractor import extract_processor_attributes
from handlers.performance import (
    handle_fp64, handle_fp64_per_core, handle_fp32, handle_fp32_per_core
)
from handlers.memory import handle_memory_bandwidth
from graph.graph_rag_builder import KNOWN_MODEL_IDS


METRIC_HANDLERS = {
    "fp64": handle_fp64,
    "fp64_per_core": handle_fp64_per_core,
    "fp32": handle_fp32,
    "fp32_per_core": handle_fp32_per_core,
    "memory": handle_memory_bandwidth
}

with open("prompts/fp_prompt.txt", "r", encoding="utf-8") as f:
    FP_PROMPT = f.read()

with open("prompts/mbw_prompt.txt", "r", encoding="utf-8") as f:
    MBW_PROMPT = f.read()

import torch

def generate_llm_response(context, question, categories, model_ids, max_tokens=1000):
    # print("[DEBUG] Entering generate_llm_response...")
    # print(f"[DEBUG] Categories: {categories}")
    # print(f"[DEBUG] Model IDs: {model_ids}")

    section_prompts = []
    results_by_category = {}

    if "memory" in categories:
        # print("[DEBUG] Processing memory category...")
        section_prompts.append(MBW_PROMPT.strip())
        results_by_category["memory"] = []
        for model_id in model_ids:
            attributes = extract_processor_attributes(context, model_id)
            # print(f"[DEBUG] Memory attributes for {model_id}: {attributes}")
            try:
                result = handle_memory_bandwidth(model_id, context, attributes)
                results_by_category["memory"].append(result)
                # print(f"[DEBUG] Memory result for {model_id}: {result}")
            except Exception as e:
                error_msg = f"⚠️ Error for {model_id} in memory: {str(e)}"
                results_by_category["memory"].append(error_msg)
                # print(f"[ERROR] {error_msg}")

    if "performance" in categories:
        # print("[DEBUG] Skipping metric classification – computing all performance metrics.")
        section_prompts.append(FP_PROMPT.strip())
        perf_metrics = ["fp64", "fp32", "fp64_per_core", "fp32_per_core"]
        results_by_category.update({cat: [] for cat in perf_metrics})
        for model_id in model_ids:
            attributes = extract_processor_attributes(context, model_id)
            # print(f"[DEBUG] Performance attributes for {model_id}: {attributes}")
            if attributes.get("fp32_units") is None and attributes.get("fp64_units") is not None:
                attributes["fp32_units"] = 2 * attributes["fp64_units"]
                # print(f"[DEBUG] Adjusted fp32_units for {model_id}: {attributes['fp32_units']}")

            for cat in perf_metrics:
                try:
                    result = METRIC_HANDLERS[cat](model_id, context, attributes)
                    results_by_category[cat].append(result)
                    # print(f"[DEBUG] {cat.upper()} result for {model_id}: {result}")
                except Exception as e:
                    error_msg = f"⚠️ Error for {model_id} in {cat}: {str(e)}"
                    results_by_category[cat].append(error_msg)
                    # print(f"[ERROR] {error_msg}")

    precomputed_blocks = []
    for category, results in results_by_category.items():
        title = category.replace("_", " ").upper()
        clean_results = [r if r else "⚠️ No result returned." for r in results]
        block = f"================== {title} ==================\n" + "\n\n".join(clean_results) + "\n=================================================="
        precomputed_blocks.append(block)

    precomputed_section = "\n\n".join(precomputed_blocks)
    print(f"[DEBUG] Precomputed section:\n{precomputed_section}")

    if categories == ["general"] or not precomputed_blocks:
        prompt = f"""
You are a highly accurate and concise processor expert. Answer the user's question using the context below.

Context:
{context}

Question:
{question}

Answer:
""".strip()
    else:
       section_text = "\n\n".join(section_prompts)

       prompt = f"""
    You are a highly accurate and concise processor expert. Use the context and the precomputed results below to answer the user's question with clarity and logic.
        
        🔹 Do NOT compute any numerical results yourself.
        🔹 Use only the precomputed values found in the sections below.
        
        {section_text}
        
        {precomputed_section}
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """.strip()

    # print(f"[DEBUG] Final Prompt:\n{prompt}")

    model = model_manager.llm_math_model
    tokenizer = model_manager.llm_math_tokenizer

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(f"[DEBUG] Decoded response:\n{decoded.strip()}")

    return decoded.strip()

def main():
    print("Initializing models. This may take a while...")
    model_manager.initialize()
    print("✅ Models initialized!")

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # Get coarse category
        categories = model_manager.classifier.get_top_coarse_categories(question)

        model_ids = extract_models_only(question, KNOWN_MODEL_IDS, model_manager.known_model_embeddings)

        context_parts = []
        if len(model_ids) >= 2:
            graph_context = get_comparison_context(model_ids)
            embedding_chunks = retrieve_chunks_for_model_ids(model_ids, question, top_k=2)
            context_parts.append(graph_context)
            context_parts.extend(embedding_chunks)
        elif len(model_ids) == 1:
            graph_context = get_graph_prompt(model_ids[0])
            embedding_chunks = retrieve_chunks_for_model_ids(model_ids, question, top_k=2)
            context_parts.append(graph_context)
            context_parts.extend(embedding_chunks)
        else:
            top_chunks = retrieve_relevant_context(question)
            context_parts.extend(top_chunks)

        context = "\n\n".join(context_parts)
        answer = generate_llm_response(context, question, categories, model_ids)
        # print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()
