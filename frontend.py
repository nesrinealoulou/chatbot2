# import requests
# import streamlit as st

# st.set_page_config(page_title="HPC Processor Chatbot", layout="wide")
# st.title("💻 HPC Processor Chatbot")
# st.markdown("Ask technical questions about AMD EPYC and Intel Xeon processors.")

# # Backend API endpoint
# API_URL = "http://localhost:8000/ask"  # Change this if your backend is elsewhere

# # Sidebar info
# st.sidebar.title("📌 Instructions")
# st.sidebar.markdown(
#     """
#     - Enter your question about processor specs (e.g., "What is the TDP and FP64 performance of 9684X?")
#     - The model will fetch structured data and compute based on verified formulas.
#     - Ensure the FastAPI server is running before using this UI.
#     """
# )

# # User input
# question = st.text_input("📝 Your Question", placeholder="e.g., What is the memory bandwidth of 9654P?")

# if st.button("Ask"):
#     if not question.strip():
#         st.warning("Please enter a valid question before asking.")
#     else:
#         with st.spinner("Processing your question..."):
#             try:
#                 response = requests.post(API_URL, json={"question": question.strip()})
#                 if response.status_code == 200:
#                     data = response.json()
#                     st.markdown("### 🧠 Answer")
#                     st.code(data.get("answer", "No answer found."), language="text")
#                 else:
#                     st.error(f"❌ Error {response.status_code}: {response.text}")
#             except Exception as e:
#                 st.error(f"⚠️ Failed to reach backend: {str(e)}")

# st.markdown("---")
# st.caption("Built with ❤️ using Streamlit + FastAPI + Transformers")
import streamlit as st

st.title("Test Input Box")

q = st.text_input("Enter a question:")

st.write(f"You asked: {q}")
