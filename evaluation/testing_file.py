import sys
from pathlib import Path
import json

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import after modifying path
from run_local_llm import run_chatbot_inference
# Load Q&A pairs
with open("q_a_db.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Run inference and collect results
test_set = []
for item in qa_data:
    question = item["question"]
    expected = item["expected_answer"]
    model_answer = run_chatbot_inference(question)

    test_set.append({
        "question": question,
        "expected_answer": expected,
        "model_answer": model_answer
    })

# Save the output
output_path = Path("final_q_a_m.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2)

print(f"✅ Test set saved to {output_path.as_posix()}")
