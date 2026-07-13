import os
import argparse
import shutil
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Restore missing GT_variation and reference images for AMZN datasets.")
    parser.add_argument(
        "--target_dirs",
        nargs="+",
        default=[
            "/root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn",
            "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/diptych",
            "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/flux-kontext",
            "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/qwen-image-edit",
            "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.25x/masked-opencv",
            "/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/noised-0.5x/masked-ocr-1.0"
        ],
        help="List of target directories to scan and restore."
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy",
        help="Path to the ground truth directory containing reference and variation images."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, lists missing files that would be copied, without actually performing the copy."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Validate ground truth directory
    if not os.path.exists(args.ground_truth_dir):
        print(f"Error: Ground truth directory '{args.ground_truth_dir}' does not exist.")
        return

    total_copies = 0

    for target_base in args.target_dirs:
        if not os.path.exists(target_base):
            print(f"Warning: Target directory '{target_base}' does not exist. Skipping.")
            continue

        print(f"\nScanning target directory: {target_base}")
        
        # Walk target_base looking for raw_meta_* categories
        for cat_name in os.listdir(target_base):
            cat_path = os.path.join(target_base, cat_name)
            if not os.path.isdir(cat_path) or not cat_name.startswith("raw_meta_"):
                continue

            gt_cat_path = os.path.join(args.ground_truth_dir, cat_name)
            if not os.path.exists(gt_cat_path):
                print(f"Warning: Category '{cat_name}' not found in ground truth. Skipping category.")
                continue

            for asin in os.listdir(cat_path):
                asin_path = os.path.join(cat_path, asin)
                if not os.path.isdir(asin_path):
                    continue

                gt_asin_path = os.path.join(gt_cat_path, asin)
                if not os.path.exists(gt_asin_path):
                    print(f"Warning: ASIN '{asin}' not found in ground truth category '{cat_name}'. Skipping.")
                    continue

                # Find any variation_<N>.jpg files in target asin path
                var_pattern = re.compile(r"^variation_(.+)\.(jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)
                
                # Check reference image
                ref_target_path = os.path.join(asin_path, "reference.jpg")
                ref_src_path = os.path.join(gt_asin_path, "reference.jpg")
                
                if os.path.exists(ref_src_path) and not os.path.exists(ref_target_path):
                    print(f"  [MISSING REFERENCE] {cat_name}/{asin}/reference.jpg")
                    total_copies += 1
                    if not args.dry_run:
                        shutil.copy(ref_src_path, ref_target_path)
                        print(f"    Restored: {ref_target_path}")

                for file_name in os.listdir(asin_path):
                    match = var_pattern.match(file_name)
                    if match:
                        suffix = match.group(1)
                        ext = match.group(2)
                        
                        # Find corresponding variation file in ground truth
                        gt_var_file_name = f"variation_{suffix}.{ext}"
                        gt_var_src_path = os.path.join(gt_asin_path, gt_var_file_name)
                        
                        # Case-insensitive search if needed
                        if not os.path.exists(gt_var_src_path):
                            for candidate in os.listdir(gt_asin_path):
                                if candidate.lower() == gt_var_file_name.lower():
                                    gt_var_src_path = os.path.join(gt_asin_path, candidate)
                                    break
                                    
                        if not os.path.exists(gt_var_src_path):
                            continue

                        # Check if either gt_variation or GT_variation exists
                        gt_target_file_1 = f"gt_variation_{suffix}.{ext}"
                        gt_target_file_2 = f"GT_variation_{suffix}.{ext}"
                        
                        target_path_1 = os.path.join(asin_path, gt_target_file_1)
                        target_path_2 = os.path.join(asin_path, gt_target_file_2)
                        
                        if not os.path.exists(target_path_1) and not os.path.exists(target_path_2):
                            dest_path = os.path.join(asin_path, gt_target_file_1)
                            print(f"  [MISSING GT VARIATION] {cat_name}/{asin}/{gt_target_file_1}")
                            total_copies += 1
                            if not args.dry_run:
                                shutil.copy(gt_var_src_path, dest_path)
                                print(f"    Restored: {dest_path}")

    print(f"\nScan completed. Total files identified/copied: {total_copies}")

if __name__ == "__main__":
    main()
