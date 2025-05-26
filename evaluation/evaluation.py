import os
import json
import evaluate
import pandas as pd
from tqdm import tqdm
from openai import OpenAI  # ✅ new SDK


# === Load Dataset ===
with open("final_q_a_m.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === Initialize scorers ===
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# === Function to call GPT-4 Judge ===
def gpt4_judge(question, expected, generated):
    prompt = f"""You are a highly specialized evaluator for processor-related questions.

Evaluate the following model-generated answer against the expected answer and the original question.

Respond with a JSON including:
- "accuracy_score" (0–5)
- "completeness_score" (0–5)
- "faithfulness_score" (0–5)
- "final_verdict": "pass" or "fail"

# QUESTION:
{question}

# EXPECTED ANSWER:
{expected}

# MODEL ANSWER:
{generated}

### JSON:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return json.loads(content.strip())
    except Exception as e:
        return {"error": str(e)}

# === Evaluate each QA pair ===
results = []
for row in tqdm(test_data, desc="Evaluating"):
    question = row["question"]
    expected = row["expected_answer"]
    generated = row["model_answer"]

    # Auto metrics
    bleu_score = bleu.compute(predictions=[generated], references=[[expected]])["bleu"]
    rouge_score = rouge.compute(predictions=[generated], references=[expected])["rougeL"]
    bert = bertscore.compute(predictions=[generated], references=[expected], lang="en")
    bert_score = sum(bert["f1"]) / len(bert["f1"])

    judge_result = gpt4_judge(question, expected, generated)

    results.append({
        "question": question,
        "expected": expected,
        "generated": generated,
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "bertscore": bert_score,
        **judge_result
    })

# Save results
# === Append to or create evaluation_results.json ===
output_file = "evaluation_results.json"

# Load existing results if the file exists
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# Append new results
existing_results.extend(results)

# Save updated results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(existing_results, f, indent=2, ensure_ascii=False)

print("✅ Evaluation results appended to evaluation_results.json")
