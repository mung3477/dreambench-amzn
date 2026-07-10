import pandas as pd
from datasets import load_dataset
import os
import json
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

def extract_images(item, img_key='large'):
    if isinstance(item, dict):
        return item.get(img_key, [])
    return []

def main():
    category = "raw_meta_All_Beauty" # We will start with a smaller dataset for testing
    print(f"Loading dataset {category}...")
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", category, trust_remote_code=True)
    df = dataset['full'].to_pandas()
    print(f"Loaded {len(df)} items.")
    
    # Keyword definitions
    keywords = {
        'text-label': ['label', 'text', 'print', 'bottle', 'packaging', 'box', 'tube'],
        'logo': ['logo', 'brand', 'emblem', 'signature'],
        'intricate-geometry': ['brush', 'machine', 'trimmer', 'clipper', 'blade']
    }
    
    # Results dictionary
    candidates = {k: [] for k in keywords.keys()}
    
    print("Filtering candidates...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row['title']).lower() if row['title'] else ""
        desc = " ".join([str(d).lower() for d in row['description']]) if isinstance(row['description'], list) else str(row['description']).lower()
        features = " ".join([str(f).lower() for f in row['features']]) if isinstance(row['features'], list) else str(row['features']).lower()
        
        combined_text = title + " " + desc + " " + features
        
        # Get images
        imgs = extract_images(row['images'], img_key='large')
        if not imgs or imgs[0] is None:
            continue
            
        # Assign to categories
        for cat, kw_list in keywords.items():
            if any(kw in combined_text for kw in kw_list):
                candidates[cat].append({
                    'title': row['title'],
                    'image_url': imgs[0], # Just taking the first large image
                    'asin': row.get('parent_asin', f"idx_{idx}")
                })
                break # Only assign to one category for now

    for cat, items in candidates.items():
        print(f"Category {cat}: {len(items)} candidates")
        
    # Save candidates to JSON for later VLM processing
    with open("candidates.json", "w") as f:
        json.dump(candidates, f, indent=4)
        
if __name__ == "__main__":
    main()
