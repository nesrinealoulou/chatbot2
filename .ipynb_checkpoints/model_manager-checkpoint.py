import time
from models.model_loader import (
    load_math_model, load_embedding_model, initialize_graph
)
from graph.graph_chunker import build_sectionwise_chunks_from_graph
from retrievers.semantic_context_retriever import build_line_index
from models.embedding_utils import (
    initialize_embedding_index, compute_known_model_embeddings
)
from prompts.classifier import IntelligentQueryClassifier
from graph.graph_rag_builder import KNOWN_MODEL_IDS

class ModelManager:
    def __init__(self):
        self.llm_math_model = None
        self.llm_math_tokenizer = None
        self.embed_model = None
        self.classifier = None
        self.known_model_embeddings = None
        self.ready = False

    def initialize(self):
        t0 = time.time()
        # Only load math model now
        self.llm_math_model, self.llm_math_tokenizer = load_math_model()
        self.embed_model = load_embedding_model()
        initialize_graph()

        chunks = build_sectionwise_chunks_from_graph()
        line_index, line_texts, line_sources = build_line_index(chunks, self.embed_model)
        initialize_embedding_index(self.embed_model, chunks)

        self.known_model_embeddings = compute_known_model_embeddings(KNOWN_MODEL_IDS)
        self.classifier = IntelligentQueryClassifier()

        self.ready = True
        print(f"✅ Models initialized in {time.time() - t0:.2f} seconds")

model_manager = ModelManager()
