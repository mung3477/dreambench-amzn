import os
import re
import json
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GROUND_TRUTH_DIR = "/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy"
TARGET_DIRS = [
    "/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/original",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/diptych",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/flux-kontext",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/qwen-image-edit",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.25x/masked-opencv",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.5x/masked-ocr-1.0"
]

def load_ground_truth_metadata():
    """Loads and compiles ground truth metadata from detail_benchmark/raw_meta_*"""
    gt_categories = {}

    if not os.path.exists(GROUND_TRUTH_DIR):
        logger.error(f"Ground truth directory not found: {GROUND_TRUTH_DIR}")
        return gt_categories

    for name in os.listdir(GROUND_TRUTH_DIR):
        full_path = os.path.join(GROUND_TRUTH_DIR, name)
        if os.path.isdir(full_path) and name.startswith('raw_meta_'):
            metadata_path = os.path.join(full_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning(f"metadata.json missing in {full_path}")
                gt_categories[name] = {'asins': set(), 'variations': {}}
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                asins = set()
                variations = {}

                for entry in data:
                    asin = entry.get('asin')
                    if not asin:
                        continue
                    asins.add(asin)

                    var_files = entry.get('variation_files', [])
                    # Extract the filename only (e.g. B08BV6F6BC/variation_1.jpg -> variation_1.jpg)
                    var_names = {os.path.basename(v.get('file')) for v in var_files if v.get('file')}
                    variations[asin] = var_names

                gt_categories[name] = {
                    'asins': asins,
                    'variations': variations
                }
            except Exception as e:
                logger.error(f"Error loading {metadata_path}: {e}")
                gt_categories[name] = {'asins': set(), 'variations': {}}

    return gt_categories

def scan_and_clean(gt_categories, execute=False):
    """Scans target directories and prints or deletes files/directories not in gt_categories"""
    total_dirs_deleted = 0
    total_files_deleted = 0

    dirs_to_delete = []
    files_to_delete = []

    for target_base in TARGET_DIRS:
        if not os.path.exists(target_base):
            logger.warning(f"Target directory does not exist: {target_base}")
            continue

        logger.info(f"Scanning target: {target_base}")

        for cat_name in os.listdir(target_base):
            cat_path = os.path.join(target_base, cat_name)
            if not os.path.isdir(cat_path) or not cat_name.startswith('raw_meta_'):
                continue

            # If the category does not exist in ground truth, mark it for entire deletion
            if cat_name not in gt_categories:
                dirs_to_delete.append(cat_path)
                continue

            gt_info = gt_categories[cat_name]
            valid_asins = gt_info['asins']
            valid_variations = gt_info['variations']

            # Scan ASIN directories in the target category
            for asin in os.listdir(cat_path):
                asin_path = os.path.join(cat_path, asin)
                if not os.path.isdir(asin_path):
                    continue

                # If ASIN is not in ground truth, delete the whole directory
                if asin not in valid_asins:
                    dirs_to_delete.append(asin_path)
                    continue

                # If ASIN is valid, scan its variation files
                allowed_vars = valid_variations.get(asin, set())
                for filename in os.listdir(asin_path):
                    file_path = os.path.join(asin_path, filename)
                    if os.path.isfile(file_path):
                        if filename == 'reference.jpg':
                            continue

                        var_match = re.match(r'^variation_(.+)\.(jpg|txt)$', filename)
                        gt_match = re.match(r'^(gt|GT)_variation_(.+)\.jpg$', filename)

                        suffix = None
                        if var_match:
                            suffix = var_match.group(1)
                        elif gt_match:
                            suffix = gt_match.group(2)

                        if suffix is not None:
                            expected_var_jpg = f"variation_{suffix}.jpg"
                            if expected_var_jpg not in allowed_vars:
                                files_to_delete.append(file_path)
                        else:
                            # If it doesn't match any pattern, remove it
                            files_to_delete.append(file_path)

    # Print dry-run summary
    print("\n" + "="*80)
    print(f"CLEANUP SUMMARY ({'EXECUTE MODE' if execute else 'DRY-RUN MODE'})")
    print("="*80)

    if dirs_to_delete:
        print(f"\nDirectories to delete ({len(dirs_to_delete)}):")
        for d in sorted(dirs_to_delete):
            print(f"  [DIR]  {d}")
    else:
        print("\nNo directories scheduled for deletion.")

    if files_to_delete:
        print(f"\nFiles to delete ({len(files_to_delete)}):")
        for f in sorted(files_to_delete):
            print(f"  [FILE] {f}")
    else:
        print("\nNo files scheduled for deletion.")

    print("-"*80)
    print(f"Total: {len(dirs_to_delete)} directories, {len(files_to_delete)} files to delete.")
    print("="*80 + "\n")

    # Execute actual deletions
    if execute:
        # Delete scheduled files
        for f in files_to_delete:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    total_files_deleted += 1
                except Exception as e:
                    logger.error(f"Failed to remove file {f}: {e}")

        # Delete scheduled directories
        for d in dirs_to_delete:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                    total_dirs_deleted += 1
                except Exception as e:
                    logger.error(f"Failed to remove directory {d}: {e}")

        # Clean up empty ASIN/category directories that might be left after file deletions
        for target_base in TARGET_DIRS:
            if not os.path.exists(target_base):
                continue
            for cat_name in os.listdir(target_base):
                cat_path = os.path.join(target_base, cat_name)
                if not os.path.isdir(cat_path) or not cat_name.startswith('raw_meta_'):
                    continue

                # Clean empty ASIN directories
                for asin in os.listdir(cat_path):
                    asin_path = os.path.join(cat_path, asin)
                    if os.path.isdir(asin_path) and not os.listdir(asin_path):
                        try:
                            os.rmdir(asin_path)
                            logger.info(f"Cleaned up empty ASIN directory: {asin_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove empty ASIN directory {asin_path}: {e}")

                # Clean empty category directories
                if os.path.exists(cat_path) and not os.listdir(cat_path):
                    try:
                        os.rmdir(cat_path)
                        logger.info(f"Cleaned up empty category directory: {cat_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove empty category directory {cat_path}: {e}")

        print(f"Successfully deleted {total_dirs_deleted} directories and {total_files_deleted} files.")

    return len(dirs_to_delete), len(files_to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up generated samples to match metadata.json")
    parser.add_argument('--execute', action='store_true', help="Execute deletions (default is dry-run)")
    args = parser.parse_args()

    gt_categories = load_ground_truth_metadata()
    scan_and_clean(gt_categories, execute=args.execute)
