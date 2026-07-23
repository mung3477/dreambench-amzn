#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
import re
import shutil
import hashlib
import argparse
from io import BytesIO
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Try importing vLLM engine
HAS_VLLM = False
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

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


def parse_json_response(response_str):
    try:
        cleaned = response_str.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]

        return json.loads(cleaned)
    except Exception as e:
        print(f"Error parsing VLM response as JSON: {e}\nResponse: {response_str}")
        return {}


def batch_generate_hf(batch_items, model, processor, max_new_tokens=256):
    """
    Batched inference using standard HuggingFace PyTorch pipeline.
    """
    batch_messages = [item["messages"] for item in batch_items]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]

    images_batch = []
    for msg in batch_messages:
        imgs = []
        for turn in msg:
            if isinstance(turn.get("content"), list):
                for elem in turn["content"]:
                    if elem.get("type") == "image":
                        img_val = elem.get("image")
                        if isinstance(img_val, str):
                            imgs.append(Image.open(img_val).convert("RGB"))
                        elif isinstance(img_val, Image.Image):
                            imgs.append(img_val)
        images_batch.append(imgs if imgs else None)

    inputs = processor(text=texts, images=images_batch, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_texts = []
    input_ids = inputs["input_ids"]
    for in_ids, out_ids in zip(input_ids, generated_ids):
        trimmed = out_ids[len(in_ids):]
        decoded = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_texts.append(decoded)

    return output_texts


def batch_generate_vllm(batch_items, vllm_engine, processor, max_new_tokens=256):
    """
    High-throughput batched inference using vLLM engine.
    """
    vllm_inputs = []
    for item in batch_items:
        messages = item["messages"]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images = []
        for turn in messages:
            if isinstance(turn.get("content"), list):
                for elem in turn["content"]:
                    if elem.get("type") == "image":
                        img_val = elem.get("image")
                        if isinstance(img_val, str):
                            images.append(Image.open(img_val).convert("RGB"))
                        elif isinstance(img_val, Image.Image):
                            images.append(img_val)

        vllm_input = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": images
            }
        }
        vllm_inputs.append(vllm_input)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=max_new_tokens
    )

    outputs = vllm_engine.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=True)
    results = [out.outputs[0].text for out in outputs]
    return results


def run_batch_inference(batch_tasks, use_vllm_engine, vllm_engine, hf_model_obj, processor, batch_size=8, max_new_tokens=256):
    if not batch_tasks:
        return []

    if use_vllm_engine:
        return batch_generate_vllm(batch_tasks, vllm_engine, processor, max_new_tokens=max_new_tokens)
    else:
        results = []
        for i in range(0, len(batch_tasks), batch_size):
            chunk = batch_tasks[i : i + batch_size]
            chunk_res = batch_generate_hf(chunk, hf_model_obj, processor, max_new_tokens=max_new_tokens)
            results.extend(chunk_res)
        return results


def main():
    parser = argparse.ArgumentParser(description="Build Amazon Review Dataset efficiently with vLLM batched inference")
    parser.add_argument("--max_candidates", type=int, default=15, help="Maximum number of candidate products to select per category")
    parser.add_argument("--category", type=str, default=None, help="Process only a specific category (e.g. raw_meta_Amazon_Fashion)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for candidate evaluation")
    parser.add_argument("--use_vllm", action="store_true", help="Force use of vLLM backend engine for inference")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="Local VLM model name/path")
    parser.add_argument("--root_dir", type=str, default="/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy", help="Output root wordy directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of model decisions")
    parser.add_argument("--verbose_dir", type=str, default="/root/Desktop/workspace/woosung/AMZN-review-2023/tmp_verbose_logs", help="Directory for verbose logs")
    args = parser.parse_args()

    root_dir = args.root_dir
    os.makedirs(root_dir, exist_ok=True)

    keywords_file = "/root/Desktop/workspace/woosung/AMZN-review-2023/category_keywords.json"
    with open(keywords_file, "r") as f:
        all_category_keywords = json.load(f)

    shutil.copy(keywords_file, os.path.join(root_dir, "category_keywords.json"))

    reference_filtering_instruction_path = "/root/Desktop/workspace/woosung/AMZN-review-2023/lib/reference_filtering_instruction.txt"
    with open(reference_filtering_instruction_path, "r") as f:
        reference_filtering_instruction = f.read().strip()

    variation_instruction_path = "/root/Desktop/workspace/woosung/AMZN-review-2023/lib/variation_filtering_instruction.txt"
    with open(variation_instruction_path, "r") as f:
        variation_prompt = f.read().strip()

    # Decide Backend Engine (vLLM vs HuggingFace PyTorch Batching)
    use_vllm_engine = HAS_VLLM and (args.use_vllm or HAS_VLLM)
    print(f"\n=== Loading Local VLM Processor: {args.model_name} ===")
    processor = AutoProcessor.from_pretrained(args.model_name)

    vllm_engine = None
    hf_model_obj = None

    if use_vllm_engine:
        print("Initializing vLLM Engine backend for high-throughput batching...")
        try:
            vllm_engine = LLM(
                model=args.model_name,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=args.gpu_memory_utilization
            )
            print("vLLM Engine initialized successfully!")
        except Exception as e:
            print(f"[WARNING] Failed to initialize vLLM engine ({e}). Falling back to PyTorch batched inference.")
            use_vllm_engine = False

    if not use_vllm_engine:
        print(f"Loading PyTorch model for batched inference (batch_size={args.batch_size})...")
        if "Qwen3" in args.model_name and Qwen3VLForConditionalGeneration is not None:
            hf_model_obj = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map="balanced"
            )
        else:
            try:
                hf_model_obj = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map="balanced", trust_remote_code=True
                )
            except Exception:
                from transformers import AutoModelImageTextToText
                hf_model_obj = AutoModelImageTextToText.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16, device_map="balanced", trust_remote_code=True
                )
        print("PyTorch model loaded successfully!")

    # Process categories
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

        filtered_metadata_path = os.path.join(category_dir, "filtered_metadata.json")
        filtered_asins = set()
        if os.path.exists(filtered_metadata_path):
            try:
                with open(filtered_metadata_path, "r", encoding="utf-8") as f:
                    filt_data = json.load(f)
                    if isinstance(filt_data, list):
                        filtered_asins = set(filt_data)
                    elif isinstance(filt_data, dict):
                        filtered_asins = set(filt_data.get("filtered_asins", []))
            except Exception as e:
                print(f"Warning: Failed to load filtered metadata: {e}")

        print("--- Phase A & B: VLM Filtering and Curation ---")
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", category_name, trust_remote_code=True, split="full", streaming=True)
        count = 0
        vlm_evaluated = 0

        # Stream candidate rows into mini-batches for parallel evaluation
        candidate_buffer = []

        for row in dataset:
            count += 1
            if count < 5:
                continue
            if count > 1000:
                break

            asin = row.get('parent_asin', f"unk_{count}")
            if asin in deleted_asins or asin in filtered_asins:
                continue

            asin_dir = os.path.join(category_dir, asin)
            if os.path.exists(asin_dir):
                if any(item.get('asin') == asin for item in final_dataset):
                    continue
                else:
                    shutil.rmtree(asin_dir, ignore_errors=True)

            imgs = extract_images(row['images'], img_key='large')
            if not imgs or len(imgs) < 2:
                filtered_asins.add(asin)
                continue

            # Download reference image
            try:
                response = requests.get(imgs[0], timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image = make_square_image(image)
                candidate_buffer.append({
                    "asin": asin,
                    "row": row,
                    "imgs": imgs,
                    "ref_image": image,
                    "asin_dir": asin_dir
                })
            except Exception:
                filtered_asins.add(asin)
                continue

            # Process buffer when batch size is reached or candidates met
            if len(candidate_buffer) >= args.batch_size or len(final_dataset) >= args.max_candidates:
                # Stage 1: Batch Detail Feature Check
                stage1_tasks = []
                for cand in candidate_buffer:
                    product_title = cand["row"]["title"]
                    msgs = [
                        {"role": "user", "content": [
                            {"type": "text", "text": reference_filtering_instruction},
                            {"type": "text", "text": f"Product Name: {product_title}"},
                            {"type": "image", "image": cand["ref_image"]},
                        ]}
                    ]
                    cand["messages_stage1"] = msgs
                    stage1_tasks.append(cand)

                stage1_responses = run_batch_inference(
                    stage1_tasks, use_vllm_engine, vllm_engine, hf_model_obj, processor, batch_size=args.batch_size, max_new_tokens=256
                )

                # Filter stage 1 accepted candidates
                stage1_accepted = []
                for cand, raw_res in zip(candidate_buffer, stage1_responses):
                    vlm_evaluated += 1
                    parsed_res = parse_json_response(raw_res)
                    if parsed_res.get("decision", "REJECT") == "ACCEPT":
                        cand["content_reference"] = parsed_res
                        stage1_accepted.append(cand)
                    else:
                        filtered_asins.add(cand["asin"])

                # Stage 2 & Stage 3: Process accepted candidates
                for cand in stage1_accepted:
                    asin = cand["asin"]
                    row = cand["row"]
                    imgs = cand["imgs"]
                    image = cand["ref_image"]
                    asin_dir = cand["asin_dir"]

                    os.makedirs(asin_dir, exist_ok=True)
                    ref_path = os.path.join(asin_dir, "reference.jpg")
                    image.save(ref_path)

                    # Build Stage 2 variation tasks
                    var_tasks = []
                    var_metadata = []
                    var_count = 0

                    for url in imgs[1:]:
                        try:
                            v_resp = requests.get(url, timeout=5)
                            v_img = Image.open(BytesIO(v_resp.content)).convert("RGB")

                            if asin in deleted_variations and deleted_variations[asin]:
                                temp_buf = BytesIO()
                                v_img.save(temp_buf, format="JPEG")
                                v_hash = hashlib.md5(temp_buf.getvalue()).hexdigest()
                                if any(var.get('hash') == v_hash for var in deleted_variations[asin]):
                                    continue

                            var_count += 1
                            msgs_var = [
                                {"role": "user", "content": [{"type": "text", "text": variation_prompt}]},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "reference image: "},
                                    {"type": "image", "image": ref_path},
                                ]},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "candidate variation image: "},
                                    {"type": "image", "image": v_img},
                                ]}
                            ]
                            var_tasks.append({"messages": msgs_var})
                            var_metadata.append({"v_img": v_img, "url": url})
                        except Exception:
                            continue

                    if not var_tasks:
                        shutil.rmtree(asin_dir, ignore_errors=True)
                        filtered_asins.add(asin)
                        continue

                    # Execute Stage 2 variation checks in batch
                    var_responses = run_batch_inference(
                        var_tasks, use_vllm_engine, vllm_engine, hf_model_obj, processor, batch_size=args.batch_size, max_new_tokens=256
                    )

                    accepted_variations_meta = []
                    for meta, raw_var_res in zip(var_metadata, var_responses):
                        parsed_var = parse_json_response(raw_var_res)
                        if parsed_var.get("decision", "REJECT") == "ACCEPT":
                            accepted_variations_meta.append(meta)

                    if not accepted_variations_meta:
                        shutil.rmtree(asin_dir, ignore_errors=True)
                        filtered_asins.add(asin)
                        continue

                    # Stage 3: Infer categorical name and subject-driven prompt
                    cat_prompt = (
                        f"Given the product title \"{row['title']}\" and the product image, "
                        "what is the specific generic categorical name of the product (e.g. 'keyboard switch', 'earrings', 'watchband', 'tote bag', 'electric lighter')? "
                        "Respond with ONLY the category name (1-3 words) in lowercase, without punctuation or articles."
                    )
                    cat_task = [{
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "image", "image": ref_path},
                                {"type": "text", "text": cat_prompt}
                            ]}
                        ]
                    }]
                    cat_res = run_batch_inference(cat_task, use_vllm_engine, vllm_engine, hf_model_obj, processor, batch_size=1, max_new_tokens=15)[0]
                    categorical_name = re.sub(r'[^a-z0-9\s-]', '', cat_res.strip().lower()).strip()
                    if not categorical_name:
                        categorical_name = "product"

                    variations = []
                    prompt_tasks = []
                    prompt_meta = []

                    for idx, meta in enumerate(accepted_variations_meta, start=1):
                        v_img = meta["v_img"]
                        v_path = os.path.join(asin_dir, f"variation_{idx}.jpg")
                        v_img.save(v_path)

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
                        msgs_prompt = [
                            {"role": "user", "content": [{"type": "text", "text": prompt_query}]},
                            {"role": "user", "content": [
                                {"type": "text", "text": "reference image: "},
                                {"type": "image", "image": ref_path},
                            ]},
                            {"role": "user", "content": [
                                {"type": "text", "text": "candidate variation image: "},
                                {"type": "image", "image": v_path},
                            ]}
                        ]
                        prompt_tasks.append({"messages": msgs_prompt})
                        prompt_meta.append(f"{asin}/variation_{idx}.jpg")
                        if len(prompt_tasks) >= 3:
                            break

                    prompt_responses = run_batch_inference(
                        prompt_tasks, use_vllm_engine, vllm_engine, hf_model_obj, processor, batch_size=args.batch_size, max_new_tokens=150
                    )

                    for var_file, prompt_res in zip(prompt_meta, prompt_responses):
                        variations.append({
                            "file": var_file,
                            "prompt": prompt_res.strip()
                        })

                    if variations:
                        final_dataset.append({
                            'asin': asin,
                            'title': row['title'],
                            'reference_file': f"{asin}/reference.jpg",
                            'categorical_name': categorical_name,
                            'variation_files': variations
                        })
                        print(f"  -> Selected item: {asin} - {row['title']} (VLM Evaluated: {vlm_evaluated})")

                candidate_buffer = []
                with open(metadata_path, "w") as f:
                    json.dump(final_dataset, f, indent=4)
                with open(filtered_metadata_path, "w") as f:
                    json.dump(list(filtered_asins), f, indent=4)

                if len(final_dataset) >= args.max_candidates:
                    print(f"Reached target candidate count of {args.max_candidates}.")
                    break

        # Save category results
        with open(metadata_path, "w") as f:
            json.dump(final_dataset, f, indent=4)
        with open(filtered_metadata_path, "w") as f:
            json.dump(list(filtered_asins), f, indent=4)

        print(f"--- Pipeline Complete for {category_name} ---")
        print(f"  Final subjects: {len(final_dataset)}")


if __name__ == "__main__":
    os.environ["greedy"] = "false"
    os.environ["top_p"] = "0.8"
    os.environ["top_k"] = "20"
    os.environ["temperature"] = "0.7"
    os.environ["repetition_penalty"] = "1.0"
    os.environ["presence_penalty"] = "1.5"
    os.environ["out_seq_length"] = "16384"
    main()
