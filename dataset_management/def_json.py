import json

def parse_definitions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the text into definition blocks
    entries = content.strip().split("------------------------------------------------------------")

    parsed_definitions = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            continue  # Skip malformed lines

        term, definition = entry.split(":", 1)
        parsed_definitions.append({
            "term": term.strip(),
            "definition": definition.strip()
        })

    return parsed_definitions

# Example usage
definitions = parse_definitions("data2/definitions.txt")

# Save to JSON file (optional)
with open("data2/definitions.json", "w", encoding="utf-8") as out_file:
    json.dump(definitions, out_file, indent=2, ensure_ascii=False)

# Print result
for item in definitions:
    print(item)
