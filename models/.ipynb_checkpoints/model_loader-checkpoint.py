import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from graph.graph_rag_builder import load_all_jsons

# Only math model now
MODEL_ID_MATH = "/home/nessrine.aloulou-ext/lustre/sw_stack-373lcd9r8io/users/nessrine.aloulou-ext/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_math_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_MATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_MATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

def initialize_graph():
    load_all_jsons()
