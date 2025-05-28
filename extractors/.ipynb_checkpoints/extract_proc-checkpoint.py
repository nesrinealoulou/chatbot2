import spacy
from rapidfuzz import process as fuzz_process
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model and known model data
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==== Load Known Model IDs ====
KNOWN_MODEL_IDS = ['9354P', '6710E', '7543', '8592+', '6768P', '9534', '6421N', '6448H', '8570', '6980P',
'9654', '8452Y', '8324P', '9365', '8471N', '9734', '6944P', '8450H', '8460H', '8558', '9462', '9475F',
'6731E', '8461V', '7713', '6730P', '6740P', '6952P', '6546P-B', '6740E', '8324PN', '6960P', '9354',
'9535', '9825', '9745', '8462Y+', '6556P-B', '9335', '6726P-B', '8458P', '6746E', '9554', '7713P',
'9655', '9480', '8480+', '7513', '8592V', '9460', '9575F', '6533P-B', '8490H', '8470', '6748P', '9845',
'6756E', '6731P', '7763', '8558U', '8534P', '6548Y+', '8581V', '6979P', '9965', '9468', '9455', '9754S',
'9565', '9634', '6548N', '6423N', '6454S', '6553P-B', '9555', '7643P', '6443N', '9470', '6745P',
'7773X', '6706P-B', '6781P', '6414U', '6787P', '9454', '8454H', '6760P', '7453', '6433NE', '8460Y+',
'6438Y+', '8470Q', '5420+', '6972P', '6554S', '6530P', '9384X', '9684X', '8468H', '6448Y', '9375F',
'6558Q', '6530', '9555P', '6458Q', '6736P', '9474F', '6741P', '7663P', '8534PN', '9554P', '6780E',
'8568Y+', '6538Y+', '9355P', '9754', '9654P', '6766E', '6767P', '5512U', '8470N', '8468V', '8434PN',
'9755', '8562Y+', '8468', '9455P', '8571N', '6563P-B', '6761P', '7573X', '9645', '6788P', '7643',
'8434P', '6428N', '6543P-B', '7543P', '9334', '8558P', '6433N', '9355', '8580', '9454P', '6747P',
'8593Q', '6438N', '75F3', '7663', '9374F', '6430', '6738P', '6737P', '5520+', '9655P', '6438M',
'6538N']

# print("🔄 Embedding model IDs...")
KNOWN_MODEL_EMBEDDINGS = {
    model_id: embed_model.encode(model_id) for model_id in KNOWN_MODEL_IDS
}
# print("✅ Model embeddings ready.")

# === NLP model for candidate extraction ===
nlp = spacy.load("en_core_web_sm")

def get_candidate_units(text: str, known_model_ids: list):
    candidates = set()

    # === Dynamic regex built from known models ===
    escaped_ids = [re.escape(mid) for mid in known_model_ids]
    known_model_regex = re.compile(r"|".join(escaped_ids), flags=re.IGNORECASE)

    # 1. Extract raw matches from full string (even if embedded)
    matches = known_model_regex.findall(text)
    candidates.update([m.upper() for m in matches])

    # 2. Also extract all substrings that *look like* model IDs (even if unknown)
    # E.g. 4 digits optionally followed by uppercase letters or symbols (no \b boundaries!)
    loose_matches = re.findall(r"\d{3,5}[A-Z+\-]*", text.upper())
    candidates.update(loose_matches)

    # 3. Remove known non-model junk (e.g., 'EPYC9334' will become '9334' if caught)
    blacklist = {"EPYC", "XEON", "CORE", "RYZEN", "CPU", "PROCESSOR", "INTEL", "AMD"}
    candidates = {c for c in candidates if c not in blacklist}

    return list(candidates)



import re

def match_to_model(candidate, known_model_ids, known_model_embeddings):
    # we try to implement all regrex from list
    candidate_clean = candidate.upper().replace("_", " ").strip()
    # print('candidate_clean' , candidate_clean) 
    # Internal thresholds
    fuzzy_threshold = 85
    cosine_threshold = 0.78

    # Fuzzy match
    best_fuzzy, fuzzy_score, _ = fuzz_process.extractOne(candidate_clean, known_model_ids)
    # print('best_fuzzy' , best_fuzzy , 'fuzzy_score' , fuzzy_score) 
    # Cosine match
    candidate_emb = embed_model.encode(candidate_clean)
    best_cosine, best_score = None, 0
    for model_id in known_model_ids:
        score = cosine_similarity([candidate_emb], [known_model_embeddings[model_id]])[0][0]
        if score > best_score:
            best_cosine, best_score = model_id, score

    # Helper: numeric core of candidate
    def extract_number(s):
        m = re.search(r'\d{3,5}', s)
        return m.group(0) if m else ""

    c_num = extract_number(candidate_clean)
    f_num = extract_number(best_fuzzy)
    cos_num = extract_number(best_cosine)

    # ✅ Final acceptance condition
    if (
        (fuzzy_score >= fuzzy_threshold and c_num == f_num and f_num in candidate_clean)
        or (best_score >= cosine_threshold and c_num == cos_num and cos_num in candidate_clean)
    ):
        print('best score',best_score)
        return best_fuzzy if fuzzy_score > best_score else best_cosine

    # 🔁 Fallback: return fuzzy top-1 match even if below threshold
    # 🔁 Fallback: retry with relaxed thresholds
    relaxed_fuzzy_threshold = 70
    relaxed_cosine_threshold = 0.70
    # print('best score',best_score)
    if (
        (fuzzy_score >= relaxed_fuzzy_threshold)
        or (best_score >= relaxed_cosine_threshold)
    ):
        return best_fuzzy if fuzzy_score > best_score else best_cosine

    # ❌ Still no valid match
    return None


def extract_models_only(question, known_model_ids, known_model_embeddings):
    candidates = get_candidate_units(question, known_model_ids)
    models = set()

    for cand in candidates:
        matched_model = match_to_model(cand, known_model_ids, known_model_embeddings)
        if matched_model:
            models.add(matched_model)

    # Refine overlapping model names
    models = list(models)
    final_models = []

    for i, m in enumerate(models):
        keep = True
        for j, n in enumerate(models):
            if i != j and m in n and m != n:
                if m not in known_model_ids or n not in known_model_ids:
                    keep = False
        if keep:
            final_models.append(m)

    return sorted(final_models)




# === MAIN CHAT LOOP ===
if __name__ == "__main__":
    print("\n🧠 Processor Model Extractor")
    print("Type your question (or 'exit' to quit):\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        models = extract_models_only(question, KNOWN_MODEL_IDS, KNOWN_MODEL_EMBEDDINGS)
        print("\n📦 Extracted Processor Models:")
        print(models)
        print("\n" + "="*60 + "\n")

