import streamlit as st
from main import (
    model_manager, extract_models_only, KNOWN_MODEL_IDS,
    search_definitions, generate_llm_response, generate_llm_for_definitions_only,
    get_comparison_context, get_graph_prompt, retrieve_chunks_for_model_ids
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "models_initialized" not in st.session_state:
    st.session_state.models_initialized = False

# Title
st.set_page_config(page_title="Processor Expert Chatbot", layout="wide")
st.title("🧠 Processor Expert Chatbot")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot can answer questions about:
    - Processor specifications and performance
    - Comparisons between different models
    - Technical definitions related to processors
    """)
    st.markdown("Example questions:")
    st.markdown("- What's the difference between Xeon Platinum 8480+ and EPYC 9654?")
    st.markdown("- Explain what SIMD means")
    st.markdown("- How much memory bandwidth does Xeon Platinum 8380 have?")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize models
if not st.session_state.models_initialized:
    with st.spinner("Initializing models. This may take a while..."):
        model_manager.initialize()
        st.session_state.models_initialized = True
    st.success("✅ All models initialized!")

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask your question about processors...")
if user_input is None:
    user_input = st.text_input("Or type your question here (fallback):")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process
    # Process
with st.spinner("Thinking..."):
    model_ids = extract_models_only(user_input, KNOWN_MODEL_IDS, model_manager.known_model_embeddings)
    context_parts = []

    # Term-level definitions
    search_result = search_definitions(user_input, top_k=3)
    grouped_defs = search_result["term_results"]
    if grouped_defs:
        for term, defs in grouped_defs.items():
            context_parts.append("\n".join([f"{d['term']}: {d['definition']}" for d in defs]))

    if len(model_ids) >= 2:
        context_parts.insert(0, get_comparison_context(model_ids))
    else:
        context_parts.insert(0, get_graph_prompt(model_ids[0]))
    context_parts.extend(retrieve_chunks_for_model_ids(model_ids, user_input, top_k=2))

    context = "\n\n".join(context_parts)
    categories = model_manager.classifier.get_top_coarse_categories(user_input)
    full_response = generate_llm_response(context, user_input, categories, model_ids)

    # ✅ Extract only each "Answer:" block from the response
    import re
    answers = re.findall(r'(?i)answer:\s*(.*?)\s*(?=\n\S|\Z)', full_response, re.DOTALL)
    if answers:
        assistant_response = "\n\n".join([f"Answer:\n{a.strip()}" for a in answers])
    else:
        assistant_response = "⚠️ No clear answer section detected in the response."

