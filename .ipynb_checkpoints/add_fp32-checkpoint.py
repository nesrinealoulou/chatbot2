import os
import json

DATA_DIR = "data2"

def update_entry(entry, file_path, index=None):
    modified = False
    specs = entry.get("specs")

    if specs:
        if "fp_units" in specs:
            specs["fp64_units"] = specs.pop("fp_units")
            modified = True
        if "fp32_units" not in specs:
            specs["fp32_units"] = 4
            modified = True
    else:
        loc = f"entry {index}" if index is not None else "top level"
        print(f"⚠️ No 'specs' field in {loc} of: {file_path}")

    return modified

def update_processor_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False

    if isinstance(data, dict):
        # Case: dict of numbered keys (like "0", "1", ...)
        for key in list(data.keys()):
            entry = data[key]
            if isinstance(entry, dict):
                if update_entry(entry, file_path, index=key):
                    modified = True
    elif isinstance(data, list):
        # Case: top-level list of entries
        for i, entry in enumerate(data):
            if isinstance(entry, dict):
                if update_entry(entry, file_path, index=i):
                    modified = True
    else:
        print(f"⚠️ Unknown structure in: {file_path}")
        return

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Updated: {file_path}")

def walk_and_update_jsons(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                update_processor_json(os.path.join(root, file))

if __name__ == "__main__":
    walk_and_update_jsons(DATA_DIR)
