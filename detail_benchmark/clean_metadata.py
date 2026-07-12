import os
import json
import argparse

def clean_metadata(base_dir, execute=False):
    """
    Traverses subdirectories under base_dir that start with 'raw_meta_'.
    Checks all metadata.json files in these subdirectories, and removes
    entries or variations where the referenced image files do not exist on disk.
    """
    for name in sorted(os.listdir(base_dir)):
        cat_dir = os.path.join(base_dir, name)
        if not os.path.isdir(cat_dir) or not name.startswith("raw_meta_"):
            continue

        metadata_path = os.path.join(cat_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        print(f"Checking {metadata_path}...")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Error reading {metadata_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"  Warning: metadata.json in {name} is not a JSON list. Skipping.")
            continue

        cleaned_data = []
        modified = False

        for entry in data:
            asin = entry.get("asin")
            ref_file = entry.get("reference_file")

            # 1. Check reference_file existence
            if ref_file:
                ref_path = os.path.join(cat_dir, ref_file)
                if not os.path.exists(ref_path):
                    print(f"  [{asin}] Reference file missing: {ref_file}. Removing entry.")
                    modified = True
                    continue
            else:
                print(f"  [{asin}] No reference_file field. Removing entry.")
                modified = True
                continue

            # 2. Check variation_files existence
            var_files = entry.get("variation_files", [])
            valid_variations = []
            for var in var_files:
                var_file = var.get("file")
                if var_file:
                    var_path = os.path.join(cat_dir, var_file)
                    if os.path.exists(var_path):
                        valid_variations.append(var)
                    else:
                        print(f"  [{asin}] Variation file missing: {var_file}. Removing variation.")
                        modified = True
                else:
                    print(f"  [{asin}] Variation missing 'file' key. Removing variation.")
                    modified = True

            if not valid_variations:
                print(f"  [{asin}] No valid variations left. Removing entry.")
                modified = True
                continue

            if len(valid_variations) != len(var_files):
                entry["variation_files"] = valid_variations
                modified = True

            cleaned_data.append(entry)

        if modified:
            if execute:
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
                    print(f"  Successfully updated {metadata_path}. (Before: {len(data)}, After: {len(cleaned_data)})")
                except Exception as e:
                    print(f"  Error writing {metadata_path}: {e}")
            else:
                print(f"  [DRY RUN] Would update {metadata_path}. (Before: {len(data)}, After: {len(cleaned_data)})")
        else:
            print(f"  No changes needed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean metadata.json by verifying referenced file existence.")
    parser.add_argument("--execute", action="store_true", help="Execute the deletion of missing metadata entries")
    args = parser.parse_args()

    # Path to the detail_benchmark directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordy")
    clean_metadata(base_dir, execute=args.execute)
