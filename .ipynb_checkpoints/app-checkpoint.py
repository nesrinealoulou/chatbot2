# app.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from model_manager import model_manager
from extractors.extract_proc import extract_models_only
from graph.graph_comparator import get_comparison_context
from graph.graph_rag_builder import get_graph_prompt
from models.embedding_utils import retrieve_chunks_for_model_ids, retrieve_relevant_context
from calculations.memory_bw import (
    extract_memory_attributes, calculate_memory_bandwidth, generate_memory_bandwidth_response
)
from handlers.performance import (
    handle_fp64, handle_fp64_per_core, handle_fp32, handle_fp32_per_core
)
from handlers.memory import handle_memory_bandwidth

app = FastAPI()

class Query(BaseModel):
    question: str

METRIC_HANDLERS = {
    "fp64": handle_fp64,
    "fp64_per_core": handle_fp64_per_core,
    "fp32": handle_fp32,
    "fp32_per_core": handle_fp32_per_core,
    "memory": handle_memory_bandwidth
}

def generate_llm_response(context: str, question: str, categories: List[str], model_ids: List[str], max_tokens: int = 512) -> str:
    # Use models loaded in model_manager
    # (Same function as before, just refer to model_manager.llm_model, etc.)

    precision_keywords = ["fp64", "fp32", "double", "single"]
    ambiguous_precision = (
        any(c in categories for c in ["fp64", "fp32", "fp64_per_core", "fp32_per_core"]) and
        not any(p in question.lower() for p in precision_keywords)
    )
    clarification_note = ""
    if ambiguous_precision:
        clarification_note = "\n\nNote: Do you want the peak double precision (FP64) or single precision (FP32) performance?"

    section_prompts = []
    use_math_model = any(c in categories for c in METRIC_HANDLERS)

    if "memory" in categories:
        from prompts import MBW_PROMPT
        section_prompts.append(MBW_PROMPT.strip())
    if any(c in categories for c in ["fp64", "fp32", "fp64_per_core", "fp32_per_core"]):
        from prompts import FP_PROMPT
        section_prompts.append(FP_PROMPT.strip())

    section_prompt = "\n\n".join(section_prompts).strip()

    results_by_category = {cat: [] for cat in categories if cat in METRIC_HANDLERS}
    for model_id in model_ids:
        attributes = extract_processor_attributes(context, model_id)

        if attributes.get("fp32_units") is None and attributes.get("fp64_units") is not None:
            attributes["fp32_units"] = 2 * attributes["fp64_units"]

        for category in results_by_category:
            try:
                result = METRIC_HANDLERS[category](model_id, context, attributes)
                results_by_category[category].append(result)
            except Exception as e:
                results_by_category[category].append(f"⚠️ Error for {model_id} in {category}: {str(e)}")

    precomputed_blocks = []
    for category, results in results_by_category.items():
        title = category.replace("_", " ").upper()
        clean_results = [r if r else "⚠️ No result returned." for r in results]
        block = f"================== {title} ==================\n" + "\n\n".join(clean_results) + "\n=================================================="
        precomputed_blocks.append(block)

    precomputed_section = "\n\n".join(precomputed_blocks)

    if categories == ["general"] or not precomputed_blocks:
        prompt = f"""
You are a highly accurate and concise processor expert. Answer the user's question using the context below.

Context:
{context}

Question: {question}

Answer:{clarification_note}
""".strip()
    else:
        prompt = f"""
You are a highly accurate and concise processor expert. Use the context and the precomputed results below to answer the user's question with clarity and logic.

🔹 Do NOT compute any numerical results yourself.
🔹 Use only the precomputed values found in the sections below.

{precomputed_section}

Context:
{context}

Question: {question}

Answer:{clarification_note}
""".strip()

    model = model_manager.llm_math_model if use_math_model else model_manager.llm_model
    tokenizer = model_manager.llm_math_tokenizer if use_math_model else model_manager.llm_tokenizer

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip() if "Answer:" in decoded else decoded.strip()

@app.on_event("startup")
async def startup_event():
    if not model_manager.ready:
        model_manager.initialize()

@app.post("/ask")
async def ask_question(query: Query):
    if not model_manager.ready:
        return {"status": "loading", "message": "Backend is still initializing. Please wait a moment."}

    question = query.question.strip()
    categories = model_manager.classifier.get_top_categories(question)
    model_ids = extract_models_only(question, model_manager.KNOWN_MODEL_IDS, model_manager.known_model_embeddings)

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
    return {"question": question, "answer": answer}
