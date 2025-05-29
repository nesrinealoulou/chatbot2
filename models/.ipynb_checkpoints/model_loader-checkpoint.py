import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from graph.graph_rag_builder import load_all_jsons
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID_MATH = "/home/nessrine.aloulou-ext/lustre/sw_stack-373lcd9r8io/users/nessrine.aloulou-ext/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"

# Optional: Use RAM-disk if available to speed up cache access
os.environ["TORCH_HOME"] = "/dev/shm"  # Only if /dev/shm has space
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

def load_math_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID_MATH,
        trust_remote_code=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_MATH,
        torch_dtype=torch.bfloat16,   # ✅ use H100-optimized dtype
        device_map="auto"             # ✅ auto map across GPU if needed
    )

    # Optional but beneficial on H100 if using PyTorch >= 2.0
    model = torch.compile(model)

    return model, tokenizer

def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

def initialize_graph():
    load_all_jsons()
