import os
import re
import json
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GROUND_TRUTH_DIR = "/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy"
RATING_BASE_DIR = "/root/Desktop/workspace/woosung/commercial-dreambench/rating/amzn"
SAMPLES_BASE_DIR = "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn"
TARGET_DIRS = [
    "/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy",
    "/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/original",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/diptych",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/flux-kontext",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/qwen-image-edit",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.25x_masked-opencv",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.5x_masked-ocr-1.0",
    "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.5x/masked-dino-adaptive"
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

def is_valid_result_key(key, gt_categories):
    """
    Validates whether a sample evaluation result key (<category>_<ASIN>_variation_<N>)
    matches ground truth metadata.
    """
    matched_cat = None
    for cat_name in sorted(gt_categories.keys(), key=len, reverse=True):
        if key.startswith(cat_name + "_"):
            matched_cat = cat_name
            break

    if not matched_cat:
        return False

    rest = key[len(matched_cat) + 1:]
    match = re.match(r'^(.+)_variation_(.+)$', rest)
    if not match:
        return False

    asin = match.group(1)
    suffix = match.group(2)

    gt_info = gt_categories[matched_cat]
    if asin not in gt_info['asins']:
        return False

    allowed_vars = gt_info['variations'].get(asin, set())
    expected_var_jpg = f"variation_{suffix}.jpg"
    return expected_var_jpg in allowed_vars

def scan_and_clean(gt_categories, execute=False):
    """Scans target directories, rating directory, and evaluation result files, printing or executing cleanup"""
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

    # Scan rating directory
    if os.path.exists(RATING_BASE_DIR):
        logger.info(f"Scanning rating target: {RATING_BASE_DIR}")
        for root, dirs, files in os.walk(RATING_BASE_DIR):
            cat_name = os.path.basename(root)
            if not cat_name.startswith('raw_meta_'):
                continue

            if cat_name not in gt_categories:
                if root not in dirs_to_delete:
                    dirs_to_delete.append(root)
                continue

            gt_info = gt_categories[cat_name]
            valid_asins = gt_info['asins']
            valid_variations = gt_info['variations']

            for filename in files:
                file_path = os.path.join(root, filename)
                rating_match = re.match(r'^(.+)_variation_(.+)\.json$', filename)
                if rating_match:
                    asin = rating_match.group(1)
                    suffix = rating_match.group(2)

                    if asin not in valid_asins:
                        files_to_delete.append(file_path)
                    else:
                        expected_var_jpg = f"variation_{suffix}.jpg"
                        allowed_vars = valid_variations.get(asin, set())
                        if expected_var_jpg not in allowed_vars:
                            files_to_delete.append(file_path)

    # Scan evaluation result JSON files (*_results.json) in samples directory
    results_files_to_update = {}  # file_path -> list of deprecated keys
    if os.path.exists(SAMPLES_BASE_DIR):
        logger.info(f"Scanning evaluation results in: {SAMPLES_BASE_DIR}")
        for root, dirs, files in os.walk(SAMPLES_BASE_DIR):
            for filename in files:
                if filename.endswith('_results.json'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        deprecated_keys = []
                        if isinstance(data, dict):
                            for key in data.keys():
                                if not is_valid_result_key(key, gt_categories):
                                    deprecated_keys.append(key)

                        if deprecated_keys:
                            results_files_to_update[file_path] = deprecated_keys
                    except Exception as e:
                        logger.error(f"Error reading evaluation result file {file_path}: {e}")

    # Scan rating summary.json files to update after deletions
    summaries_to_update = {}  # summary_file -> dict of updated fields
    if os.path.exists(RATING_BASE_DIR):
        logger.info(f"Scanning rating summary files in: {RATING_BASE_DIR}")
        files_to_delete_set = set(files_to_delete)
        for root, dirs, files in os.walk(RATING_BASE_DIR):
            if 'summary.json' in files:
                summary_path = os.path.join(root, 'summary.json')
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)

                    # Collect remaining valid rating json scores under this directory
                    scores = []
                    valid_rating_files = []
                    for r, d, fs in os.walk(root):
                        for file in fs:
                            if file != 'summary.json' and file.endswith('.json'):
                                fpath = os.path.join(r, file)
                                rating_match = re.match(r'^(.+)_variation_(.+)\.json$', file)
                                if rating_match and fpath not in files_to_delete_set and os.path.exists(fpath):
                                    valid_rating_files.append(fpath)

                    valid_rating_files.sort()
                    for fpath in valid_rating_files:
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                rf_data = json.load(f)
                            if 'score' in rf_data:
                                scores.append(rf_data['score'])
                        except Exception as e:
                            logger.error(f"Error reading rating file {fpath}: {e}")

                    new_total = len(scores)
                    new_mean = sum(scores) / new_total if new_total > 0 else 0.0

                    old_total = summary_data.get('total_pairs_evaluated', 0)
                    old_mean = summary_data.get('mean_score', 0.0)

                    summaries_to_update[summary_path] = {
                        'summary_data': summary_data,
                        'new_scores': scores,
                        'new_total': new_total,
                        'new_mean': new_mean,
                        'old_total': old_total,
                        'old_mean': old_mean
                    }
                except Exception as e:
                    logger.error(f"Error reading summary file {summary_path}: {e}")

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

    if results_files_to_update:
        total_dep_keys = sum(len(keys) for keys in results_files_to_update.values())
        print(f"\nEvaluation result files to update ({len(results_files_to_update)} files, {total_dep_keys} deprecated items):")
        for f, keys in sorted(results_files_to_update.items()):
            print(f"  [RESULTS] {f} ({len(keys)} deprecated keys to remove)")
    else:
        print("\nNo evaluation result files have deprecated keys.")

    if summaries_to_update:
        print(f"\nRating summary files to sync ({len(summaries_to_update)} files):")
        for s_path, s_info in sorted(summaries_to_update.items()):
            print(f"  [SUMMARY] {s_path}: total_pairs ({s_info['old_total']} -> {s_info['new_total']}), mean_score ({s_info['old_mean']:.4f} -> {s_info['new_mean']:.4f})")
    else:
        print("\nNo rating summary files to sync.")

    print("-"*80)
    total_dep_keys = sum(len(keys) for keys in results_files_to_update.values())
    print(f"Total: {len(dirs_to_delete)} dirs to delete, {len(files_to_delete)} files to delete, {total_dep_keys} evaluation result keys to remove, {len(summaries_to_update)} summaries to sync.")
    print("="*80 + "\n")

    # Execute actual deletions and updates
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

        # Update evaluation result JSON files
        total_eval_keys_removed = 0
        for fpath, dep_keys in results_files_to_update.items():
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for k in dep_keys:
                        data.pop(k, None)
                    with open(fpath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    total_eval_keys_removed += len(dep_keys)
                    logger.info(f"Removed {len(dep_keys)} deprecated keys from {fpath}")
                except Exception as e:
                    logger.error(f"Failed to update evaluation result file {fpath}: {e}")

        # Update rating summary files
        total_summaries_updated = 0
        for s_path, s_info in summaries_to_update.items():
            if os.path.exists(s_path):
                try:
                    s_data = s_info['summary_data']
                    s_data['total_pairs_evaluated'] = s_info['new_total']
                    s_data['mean_score'] = s_info['new_mean']
                    s_data['scores'] = s_info['new_scores']
                    # Latencies and mean_latency_sec are preserved unchanged
                    with open(s_path, 'w', encoding='utf-8') as f:
                        json.dump(s_data, f, indent=4, ensure_ascii=False)
                    total_summaries_updated += 1
                    logger.info(f"Updated rating summary file {s_path}")
                except Exception as e:
                    logger.error(f"Failed to update summary file {s_path}: {e}")

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

        # Clean up empty rating directories recursively bottom-up
        if os.path.exists(RATING_BASE_DIR):
            for root, dirs, files in os.walk(RATING_BASE_DIR, topdown=False):
                if os.path.exists(root) and not os.listdir(root):
                    try:
                        os.rmdir(root)
                        logger.info(f"Cleaned up empty rating directory: {root}")
                    except Exception as e:
                        logger.warning(f"Could not remove empty rating directory {root}: {e}")

        print(f"Successfully deleted {total_dirs_deleted} directories, {total_files_deleted} files, removed {total_eval_keys_removed} evaluation result keys, and updated {total_summaries_updated} summary files.")

    return len(dirs_to_delete), len(files_to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up generated samples to match metadata.json")
    parser.add_argument('--execute', action='store_true', help="Execute deletions (default is dry-run)")
    args = parser.parse_args()

    gt_categories = load_ground_truth_metadata()
    scan_and_clean(gt_categories, execute=args.execute)

