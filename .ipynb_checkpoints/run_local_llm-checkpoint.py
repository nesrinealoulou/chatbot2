# smart_qa_rag.py

import os
import json
import re
import torch
import faiss
from transformers import pipeline
import networkx as nx
import numpy as np
from semantic_context_retriever import build_line_index, search_semantic_lines
from typing import Dict, Tuple, Union
from calculations import extract_processor_attributes, calculate_fp64_performance, generate_fp64_response

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from graph_rag_builder import (
    load_all_jsons,
    get_graph_prompt,
    get_subgraph_for_model,
    KNOWN_MODEL_IDS,
    G
)
from extract_proc import extract_models_only
from classifier import SmartQueryClassifier 

with open("prompts/fp_prompt.txt", "r", encoding="utf-8") as f:
    FP_PROMPT = f.read()

with open("prompts/mbw_prompt.txt", "r", encoding="utf-8") as f:
    MBW_PROMPT = f.read()

DATA_FOLDER = "data2"
MODEL_ID = "/home/nessrine.aloulou-ext/lustre/sw_stack-373lcd9r8io/users/nessrine.aloulou-ext/models--NousResearch--Nous-Hermes-2-Mistral-7B-DPO/snapshots/ebec0a691037d38955727d6949798429a63929dd"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID_MATH = "/home/nessrine.aloulou-ext/lustre/sw_stack-373lcd9r8io/users/nessrine.aloulou-ext/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"

llm_math_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_MATH, trust_remote_code=True)
llm_math_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_MATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

embed_model = SentenceTransformer(EMBED_MODEL)
load_all_jsons()

from graph_rag_builder import build_sectionwise_chunks_from_graph
chunks = build_sectionwise_chunks_from_graph()
line_index, line_texts, line_sources = build_line_index(chunks, embed_model)
chunk_texts = [c["text"] for c in chunks]
chunk_embeddings = embed_model.encode(chunk_texts, convert_to_tensor=False)
index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
index.add(np.array(chunk_embeddings))



def compute_known_model_embeddings(known_model_ids):
    model_id_list = list(known_model_ids)
    embeddings = embed_model.encode(model_id_list, convert_to_tensor=False)
    return {model_id: emb for model_id, emb in zip(model_id_list, embeddings)}
    
def retrieve_chunks_for_model_ids(model_ids: list[str], question: str, top_k=3):
    model_chunks = [c for c in chunks if c["model_id"] in model_ids]
    texts = [c["text"] for c in model_chunks]
    embeddings = embed_model.encode(texts, convert_to_tensor=False)
    index_local = faiss.IndexFlatL2(len(embeddings[0]))
    index_local.add(np.array(embeddings))

    q_emb = embed_model.encode(question, convert_to_tensor=False)
    D, I = index_local.search(np.array([q_emb]), top_k)
    return [texts[i] for i in I[0]]

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
    
def generate_llm_response(context: str, question: str, max_tokens: int = 512) -> str:
    categories = classifier.get_top_categories(question)
    section_prompts = []
    use_math_model = False

    if "memory" in categories:
        section_prompts.append(MBW_PROMPT.strip())
        use_math_model = True
    if "performance" in categories:
        section_prompts.append(FP_PROMPT.strip())
        use_math_model = True

    fp64_calculated_text = ""
    model_ids = extract_models_only(question, KNOWN_MODEL_IDS, compute_known_model_embeddings(KNOWN_MODEL_IDS))
    for model_id in model_ids:
        attributes = extract_processor_attributes(context, model_id)
        results, error = calculate_fp64_performance(attributes)
        if not error:
            fp64_calculated_text += generate_fp64_response(model_id, results) + "\n\n"

    section_prompt = "\n\n".join(section_prompts)

    main_prompt = f"""
You are a highly accurate and concise processor expert. Use the context and the precomputed results below to answer the user's question with clarity and logic.

🔹 Do NOT compute any numerical results yourself.
🔹 Use only the precomputed values found in the FP64 RESULTS section.

================== FP64 RESULTS ==================
{fp64_calculated_text.strip()}
==================================================

Context:
{context}

Question: {question}

Answer:
""".strip()

    full_prompt = f"{section_prompt}\n\n{main_prompt}" if section_prompt else main_prompt
    model, tokenizer = (llm_math_model, llm_math_tokenizer) if use_math_model else (llm_model, llm_tokenizer)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip() if "Answer:" in decoded else decoded.strip()



def run_chatbot_inference(question: str) -> dict:
    """
    Given a question, run full pipeline:
    1. Extract model IDs
    2. Retrieve context (graph + semantic chunks)
    3. Generate answer
    Returns a dictionary with all relevant info.
    """
    model_ids = extract_models_only(question, KNOWN_MODEL_IDS, compute_known_model_embeddings(KNOWN_MODEL_IDS))

    context_parts = []

    # Case A: Multiple models (e.g., comparison)
    if len(model_ids) >= 2:
        graph_context = get_comparison_context(model_ids)
        embedding_chunks = retrieve_chunks_for_model_ids(model_ids, question, top_k=2)
        context_parts.append(graph_context)
        context_parts.extend(embedding_chunks)

    # Case B: Single model
    elif len(model_ids) == 1:
        graph_context = get_graph_prompt(model_ids[0])
        embedding_chunks = retrieve_chunks_for_model_ids(model_ids, question, top_k=2)
        context_parts.append(graph_context)
        context_parts.extend(embedding_chunks)

    # Case C: No model detected — fallback
    else:
        top_chunks = retrieve_relevant_context(question)
        context_parts.extend(top_chunks)

    context = "\n\n".join(context_parts)
    response = generate_llm_response(context, question)

    return response

if __name__ == "__main__":
    print("🧠 Processor Chatbot is ready. Type your question or 'exit' to quit.\n")
    known_model_embeddings = compute_known_model_embeddings(KNOWN_MODEL_IDS)

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # 🧠 Extract model IDs 
        model_ids = extract_models_only(question, KNOWN_MODEL_IDS, known_model_embeddings)
        print("📦 Models Detected:", model_ids)
        

        context_parts = []

        # Case A: Multiple models (e.g., comparison)
        if len(model_ids) >= 2:
            graph_context = get_comparison_context(model_ids)
            embedding_chunks = retrieve_chunks_for_model_ids(model_ids, question, top_k=2)
            context_parts.append(graph_context)
            context_parts.extend(embedding_chunks)

        # Case B: Single model
        elif len(model_ids) == 1:
            graph_context = get_graph_prompt(model_ids[0])
            embedding_chunks = retrieve_chunks_for_model_ids(model_ids, question, top_k=2)
            context_parts.append(graph_context)
            context_parts.extend(embedding_chunks)

        # Case C: No model, fallback to similarity search
        else:
            top_chunks = retrieve_relevant_context(question)
            context_parts.extend(top_chunks)

        # 🧾 Final context assembly
        # 🧾 Final context assembly
        context = "\n\n".join(context_parts)
        
        # Print context
        print("\n🔍 Context used for answering:")
        print("=" * 60)
        # print(context)
        
        # Separator before the answer
        print("\n🧠 Answer:\n" + "=" * 60)

        classifier = SmartQueryClassifier(llm_model, llm_tokenizer)
        # Generate and print the answer
        print(generate_llm_response(context, question))

