import os
import json
import time
import requests
import re
import shutil
import argparse
from io import BytesIO
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

def make_square_image(image, fill_color=(255, 255, 255)):
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        new_image = Image.new("RGB", (width, width), fill_color)
        top_offset = (width - height) // 2
        new_image.paste(image, (0, top_offset))
        return new_image
    else:
        new_image = Image.new("RGB", (height, height), fill_color)
        left_offset = (height - width) // 2
        new_image.paste(image, (left_offset, 0))
        return new_image

def extract_images(item, img_key='large'):
    keys_to_try = ['hi_res', img_key]
    if isinstance(item, dict):
        for k in keys_to_try:
            val = item.get(k, [])
            if val:
                if isinstance(val, list):
                    filtered = [v for v in val if v]
                    if filtered:
                        return filtered
                elif isinstance(val, str):
                    return [val]
        return []
    elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
        res = []
        for i in item:
            for k in keys_to_try:
                val = i.get(k, None)
                if val:
                    if isinstance(val, list):
                        filtered = [v for v in val if v]
                        if filtered:
                            res.extend(filtered)
                            break
                    elif isinstance(val, str):
                        res.append(val)
                        break
        return res
    return []



def generate_with_vlm(messages, model, processor, max_new_tokens=32):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def parse_json_response(response_str):
    try:
        # Clean markdown code block formatting if present
        cleaned = response_str.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Locate the outermost JSON object
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]

        return json.loads(cleaned)
    except Exception as e:
        print(f"Error parsing VLM response as JSON: {e}\nResponse: {response_str}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Build Amazon Review Dataset")
    parser.add_argument("--max_candidates", type=int, default=15, help="Maximum number of candidate products to select per category")
    parser.add_argument("--category", type=str, default=None, help="Process only a specific category (e.g. raw_meta_Amazon_Fashion)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of model decisions and images to a temp folder")
    parser.add_argument("--verbose_dir", type=str, default="/root/Desktop/workspace/woosung/AMZN-review-2023/tmp_verbose_logs", help="Directory where verbose logs will be saved")
    args = parser.parse_args()

    root_dir = "/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy"
    os.makedirs(root_dir, exist_ok=True)

    # Load dynamic keywords to know what categories exist
    keywords_file = "/root/Desktop/workspace/woosung/AMZN-review-2023/category_keywords.json"
    with open(keywords_file, "r") as f:
        all_category_keywords = json.load(f)

    # Copy keywords to output dir for documentation
    shutil.copy(keywords_file, os.path.join(root_dir, "category_keywords.json"))

    # Load the reference filtering instruction prompt
    reference_filtering_instruction_path = "/root/Desktop/workspace/woosung/AMZN-review-2023/lib/reference_filtering_instruction.txt"
    with open(reference_filtering_instruction_path, "r") as f:
        reference_filtering_instruction = f.read().strip()

    # Load the variation instruction prompt
    variation_instruction_path = "/root/Desktop/workspace/woosung/AMZN-review-2023/lib/variation_filtering_instruction.txt"
    with open(variation_instruction_path, "r") as f:
        variation_prompt = f.read().strip()



    # Pre-load VLM (Do this once to save memory and time)
    print("\n=== Pre-loading VLM Model ===")
    model_name = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Loading local VLM model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="balanced", trust_remote_code=True
        )
    except Exception as e:
        print("Fallback model loading...")
        if Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="balanced", trust_remote_code=True
            )
        else:
            raise e

    # Loop over all 32 categories
    for category_name, KEYWORDS in all_category_keywords.items():
        if args.category and category_name != args.category:
            continue

        print(f"\n==============================================")
        print(f"Processing Category: {category_name}")
        print(f"==============================================")

        category_dir = os.path.join(root_dir, category_name)
        metadata_path = os.path.join(category_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    final_dataset = json.load(f)
            except Exception:
                final_dataset = []
        else:
            final_dataset = []

        if len(final_dataset) >= args.max_candidates:
            print(f"  -> Skipping {category_name}: Already has {len(final_dataset)} candidates.")
            continue

        os.makedirs(category_dir, exist_ok=True)

        # Load deleted metadata if exists
        deleted_metadata_path = os.path.join(category_dir, "deleted_metadata.json")
        deleted_asins = {}
        deleted_variations = {}
        if os.path.exists(deleted_metadata_path):
            try:
                with open(deleted_metadata_path, "r", encoding="utf-8") as f:
                    del_data = json.load(f)
                    deleted_asins = del_data.get("deleted_asins", {})
                    deleted_variations = del_data.get("deleted_variations", {})
            except Exception as e:
                print(f"Warning: Failed to load deleted metadata: {e}")

        print("--- Phase A & B: VLM Filtering and Curation ---")
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", category_name, trust_remote_code=True, split="full", streaming=True)
        count = 0
        vlm_evaluated = 0

        for row in dataset:
            count += 1
            if count < 5:
                continue
            if count > 1000: # Limit candidate search space per category to 5000 items (dry-run/scale limit)
                break

            asin = row.get('parent_asin', f"unk_{count}")
            if asin in deleted_asins:
                print(f"  -> Skipping ASIN {asin}: Marked as deleted in deleted_metadata.json.")
                continue

            asin_dir = os.path.join(category_dir, asin)
            if os.path.exists(asin_dir):
                if any(item.get('asin') == asin for item in final_dataset):
                    print(f"  -> Skipping ASIN {asin}: Already processed.")
                    continue
                else:
                    # Clean up orphaned directory
                    shutil.rmtree(asin_dir, ignore_errors=True)

            imgs = extract_images(row['images'], img_key='large')
            if not imgs or len(imgs) < 2:
                continue

            try:
                # Download reference image
                response = requests.get(imgs[0], timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image = make_square_image(image)

                tmp_path = os.path.join(root_dir, f"tmp_{category_name}.jpg")
                image.save(tmp_path)

                vlm_evaluated += 1

                product_title = row['title']
                item_verbose_dir = None
                if args.verbose:
                    item_verbose_dir = os.path.join(args.verbose_dir, category_name, f"{vlm_evaluated:04d}_{asin}")
                    os.makedirs(item_verbose_dir, exist_ok=True)
                    # Save reference image that the model saw
                    image.save(os.path.join(item_verbose_dir, "reference.jpg"))

                # Stage 1: Detail Feature Check using filtering_instruction.txt
                messages_detail = [
                    {"role": "user", "content": [
                        {"type": "text", "text": reference_filtering_instruction},
                        {"type": "text", "text": f"Product Name: {product_title}"},
                        {"type": "image", "image": f"{tmp_path}"},
                    ]}
                ]
                content_reference_raw = generate_with_vlm(messages_detail, model, processor, max_new_tokens=256)
                content_reference = parse_json_response(content_reference_raw)

                if item_verbose_dir:
                    with open(os.path.join(item_verbose_dir, "stage1_reference_prompt.txt"), "w") as f:
                        f.write(reference_filtering_instruction)
                    with open(os.path.join(item_verbose_dir, "stage1_reference_response_raw.txt"), "w") as f:
                        f.write(content_reference_raw)
                    with open(os.path.join(item_verbose_dir, "stage1_reference_response_parsed.json"), "w") as f:
                        json.dump(content_reference, f, indent=4)

                # Check if the reference image contains sufficient fine-grained details for subject identity preservation.
                if content_reference.get("decision", "REJECT") != "ACCEPT":
                    continue


                # Stage 2: Variation Check (Less strict comparison with the first/reference image)
                os.makedirs(asin_dir, exist_ok=True)

                ref_path = os.path.join(asin_dir, "reference.jpg")
                image.save(ref_path)

                categorical_name = None
                variations = []
                var_evaluated_count = 0
                for url in imgs[1:]:
                    try:
                        v_resp = requests.get(url, timeout=5)
                        v_img = Image.open(BytesIO(v_resp.content)).convert("RGB")

                        # Check against deleted variations
                        if asin in deleted_variations and deleted_variations[asin]:
                            import hashlib
                            temp_buf = BytesIO()
                            v_img.save(temp_buf, format="JPEG")
                            v_hash = hashlib.md5(temp_buf.getvalue()).hexdigest()
                            if any(var.get('hash') == v_hash for var in deleted_variations[asin]):
                                print(f"  -> Skipping variation for ASIN {asin}: Matches a deleted variation in deleted_metadata.json.")
                                continue

                        v_tmp_path = os.path.join(root_dir, f"tmp_v_{category_name}.jpg")
                        v_img.save(v_tmp_path)

                        var_evaluated_count += 1
                        if item_verbose_dir:
                            # Save the variation image that the model saw
                            v_img.save(os.path.join(item_verbose_dir, f"variation_{var_evaluated_count}.jpg"))

                        # Stage 2: Variation Validation Check
                        messages_var = [
                            {"role": "user", "content": [
                                {"type": "text", "text": variation_prompt}
                            ]},
                            {"role": "user", "content": [
                                {"type": "text", "text": "reference image: "},
                                {"type": "image", "image": f"{ref_path}"},
                            ]},
                            {"role": "user", "content": [
                                {"type": "text", "text": "candidate variation image: "},
                                {"type": "image", "image": f"{v_tmp_path}"},
                            ]}
                        ]
                        content_var_raw = generate_with_vlm(messages_var, model, processor, max_new_tokens=256)
                        content_var = parse_json_response(content_var_raw)

                        if item_verbose_dir:
                            with open(os.path.join(item_verbose_dir, f"stage2_var_prompt_{var_evaluated_count}.txt"), "w") as f:
                                f.write(variation_prompt)
                            with open(os.path.join(item_verbose_dir, f"stage2_var_response_raw_{var_evaluated_count}.txt"), "w") as f:
                                f.write(content_var_raw)
                            with open(os.path.join(item_verbose_dir, f"stage2_var_response_parsed_{var_evaluated_count}.json"), "w") as f:
                                json.dump(content_var, f, indent=4)

                        if content_var.get("decision", "REJECT") == "ACCEPT":
                            if categorical_name is None:
                                # Infer category only when a variation passes filtering
                                cat_prompt = (
                                    f"Given the product title \"{row['title']}\" and the product image, "
                                    "what is the specific generic categorical name of the product (e.g. 'keyboard switch', 'earrings', 'watchband', 'tote bag', 'electric lighter')? "
                                    "Respond with ONLY the category name (1-3 words) in lowercase, without punctuation or articles."
                                )
                                messages_cat = [
                                    {"role": "user", "content": [
                                        {"type": "image", "image": f"{ref_path}"},
                                        {"type": "text", "text": cat_prompt}
                                    ]}
                                ]
                                categorical_name = generate_with_vlm(messages_cat, model, processor, max_new_tokens=15).strip().lower()
                                categorical_name = re.sub(r'[^a-z0-9\s-]', '', categorical_name).strip()
                                if not categorical_name:
                                    categorical_name = "product"

                            idx = len(variations) + 1
                            v_path = os.path.join(asin_dir, f"variation_{idx}.jpg")
                            v_img.save(v_path)

                            # Query VLM using standard prompt extraction system instruction
                            prompt_query = (
                                "You are an expert in writing prompts for Subject Driven Image Generation (like DreamBooth, Flux-ControlNet, etc.).\n"
                                f"Given a reference image of the product (Image 1) and a target advertisement/infographic image (Image 2), "
                                f"write a text prompt describing the target image (Image 2) so a generative model can reproduce its style and layout with a custom subject.\n\n"
                                "Rules:\n"
                                f"1. Use the generic class name 'the {categorical_name}' (or 'the product') for the subject instead of brand-specific names.\n"
                                f"2. Do NOT describe the intrinsic visual properties of the product itself (such as its specific colors, original brand logo printings, or specific text/labels printed directly on the product's body). Never use phrases like 'with its original design and branding visible' or 'showing its original design'. Leave the subject's visual appearance completely undescribed, as those details must come solely from the custom subject itself.\n"
                                f"3. CRITICAL: Completely ignore and omit any text, words, headlines, slogans, numbers, disclaimers, or copies present in Image 2. Do NOT describe them, do NOT quote them, and do NOT describe placeholder shapes like blank boxes or outlines representing text. Describe the background and panels simply as solid-colored, plain, or empty (e.g. 'a solid yellow background panel', 'a plain white background', 'a solid peach-pink background'). All text and placeholder elements must be completely ignored as if they do not exist, leaving the layout clean, plain, and natural.\n"
                                f"4. Describe the layout, environment, lighting, action, and main external graphic/color panels precisely (e.g. 'A split-screen layout...', 'A product shot on a plain background...').\n"
                                f"5. Keep the description natural, clear, and direct. Do NOT start with 'The target image...' or 'In Image 2...'. Describe the image layout and content directly."
                            )
                            messages_prompt = [
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt_query}
                                ]},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "reference image: "},
                                    {"type": "image", "image": f"{ref_path}"},
                                ]},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "candidate variation image: "},
                                    {"type": "image", "image": f"{v_tmp_path}"},
                                ]}
                            ]
                            variation_prompt_str = generate_with_vlm(messages_prompt, model, processor, max_new_tokens=150).strip()

                            if item_verbose_dir:
                                with open(os.path.join(item_verbose_dir, f"stage2_caption_prompt_{var_evaluated_count}.txt"), "w") as f:
                                    f.write(prompt_query)
                                with open(os.path.join(item_verbose_dir, f"stage2_caption_response_{var_evaluated_count}.txt"), "w") as f:
                                    f.write(variation_prompt_str)

                            variations.append({
                                "file": f"{asin}/variation_{idx}.jpg",
                                "prompt": variation_prompt_str
                            })

                            if len(variations) >= 3:
                                break
                    except Exception as e:
                        if item_verbose_dir:
                            try:
                                with open(os.path.join(item_verbose_dir, f"error_variation_{var_evaluated_count}.txt"), "w") as f:
                                    f.write(f"Exception raised in variation check: {str(e)}")
                            except Exception:
                                pass
                        pass

                if len(variations) > 0:
                    final_dataset.append({
                        'asin': asin,
                        'title': row['title'],
                        'reference_file': f"{asin}/reference.jpg",
                        'categorical_name': categorical_name,
                        'variation_files': variations
                    })
                    print(f"  -> Selected item: {asin} - {row['title']} (VLM Evaluated: {vlm_evaluated})")
                    if len(final_dataset) >= args.max_candidates:
                        print(f"Reached target candidate count of {args.max_candidates}.")
                        break
                else:
                    shutil.rmtree(asin_dir, ignore_errors=True)
            except Exception as e:
                if args.verbose and 'item_verbose_dir' in locals() and item_verbose_dir:
                    try:
                        with open(os.path.join(item_verbose_dir, "error_main.txt"), "w") as f:
                            f.write(f"Exception raised in main product loop: {str(e)}")
                    except Exception:
                        pass
                pass

        # Phase C: Final output for this category
        # Clean up temporary files
        tmp_path = os.path.join(root_dir, f"tmp_{category_name}.jpg")
        v_tmp_path = os.path.join(root_dir, f"tmp_v_{category_name}.jpg")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(v_tmp_path):
            os.remove(v_tmp_path)

        metadata_path = os.path.join(category_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(final_dataset, f, indent=4)

        print(f"--- Pipeline Complete for {category_name} ---")
        print(f"  Final subjects: {len(final_dataset)}")

if __name__ == "__main__":
    # Standard generation environment variables as set in compositional_reasoning.ipynb
    os.environ["greedy"] = "false"
    os.environ["top_p"] = "0.8"
    os.environ["top_k"] = "20"
    os.environ["temperature"] = "0.7"
    os.environ["repetition_penalty"] = "1.0"
    os.environ["presence_penalty"] = "1.5"
    os.environ["out_seq_length"] = "16384"
    main()
