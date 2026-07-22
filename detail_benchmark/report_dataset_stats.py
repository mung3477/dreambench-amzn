#!/usr/bin/env python3
import os
import json
import argparse
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Report statistics of a dataset directory and generate visualizations.")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="/root/Desktop/workspace/woosung/AMZN-review-2023/detail_benchmark/wordy",
        help="Path to the dataset directory (e.g. wordy)"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        default=None,
        help="Path to output JSON stats file (defaults to <dataset_dir>/dataset_stats.json)"
    )
    parser.add_argument(
        "--output_html", 
        type=str, 
        default=None,
        help="Path to output HTML visualization file (defaults to <dataset_dir>/dataset_stats.html)"
    )
    return parser.parse_args()

def clean_category_name(raw_name: str) -> str:
    """Converts raw_meta_Category_Name to Category Name"""
    name = raw_name
    if name.startswith("raw_meta_"):
        name = name[len("raw_meta_"):]
    return name.replace("_", " ")

def generate_statistics(dataset_dir: str) -> Dict[str, Any]:
    """Scans the dataset directory and computes statistics."""
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    stats = {
        "dataset_directory": os.path.abspath(dataset_dir),
        "total_categories": 0,
        "total_subjects": 0,
        "total_variations": 0,
        "categories": {}
    }

    # Find raw_meta_ directories
    category_dirs = []
    for entry in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("raw_meta_"):
            category_dirs.append(entry)
    
    category_dirs.sort()
    stats["total_categories"] = len(category_dirs)

    for cat_dir in category_dirs:
        cat_path = os.path.join(dataset_dir, cat_dir)
        metadata_path = os.path.join(cat_path, "metadata.json")
        
        subjects_data = []
        num_subjects = 0
        num_variations = 0
        categorical_names = set()
        representative_samples = []

        # Read metadata.json if it exists
        metadata_exists = os.path.exists(metadata_path)
        if metadata_exists:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)
                
                for item in metadata_list:
                    asin = item.get("asin")
                    if not asin:
                        continue
                    
                    num_subjects += 1
                    var_files = item.get("variation_files", [])
                    num_variations += len(var_files)
                    
                    cat_name = item.get("categorical_name")
                    if cat_name:
                        categorical_names.add(cat_name)
                    
                    # Store details for visualization
                    # reference_file is typically like "ASIN/reference.jpg"
                    # We want to make it relative to the dataset_dir: cat_dir + "/" + reference_file
                    ref_rel = item.get("reference_file")
                    if ref_rel:
                        ref_path_from_dataset = os.path.join(cat_dir, ref_rel)
                    else:
                        ref_path_from_dataset = os.path.join(cat_dir, asin, "reference.jpg")
                    
                    subjects_data.append({
                        "asin": asin,
                        "title": item.get("title", ""),
                        "categorical_name": cat_name or "",
                        "reference_image": ref_path_from_dataset,
                        "num_variations": len(var_files),
                        "variation_prompts": [v.get("prompt", "") for v in var_files]
                    })
            except Exception as e:
                logger.error(f"Error reading metadata.json in {cat_path}: {e}")
                metadata_exists = False

        # Fallback/validation from disk
        if not metadata_exists or num_subjects == 0:
            # Let's scan folders on disk
            for entry in os.listdir(cat_path):
                sub_path = os.path.join(cat_path, entry)
                # Subject is a directory (excluding deleted_metadata.json and system dirs)
                if os.path.isdir(sub_path) and not entry.startswith("."):
                    asin = entry
                    num_subjects += 1
                    
                    # Look for variation images
                    vars_on_disk = []
                    ref_img = None
                    for file_entry in os.listdir(sub_path):
                        if file_entry.startswith("variation_") and file_entry.lower().endswith((".jpg", ".png", ".jpeg")):
                            vars_on_disk.append(file_entry)
                        elif file_entry.lower() == "reference.jpg" or file_entry.lower() == "reference.png":
                            ref_img = file_entry
                    
                    num_variations += len(vars_on_disk)
                    
                    subjects_data.append({
                        "asin": asin,
                        "title": f"Subject {asin}",
                        "categorical_name": "unknown",
                        "reference_image": os.path.join(cat_dir, asin, ref_img or "reference.jpg"),
                        "num_variations": len(vars_on_disk),
                        "variation_prompts": []
                    })

        # Collect representative images (e.g. first 3 subjects' references)
        representative_samples = []
        for s in subjects_data[:3]:
            # Verify if file actually exists
            img_path = os.path.join(dataset_dir, s["reference_image"])
            if os.path.exists(img_path):
                representative_samples.append({
                    "asin": s["asin"],
                    "categorical_name": s["categorical_name"],
                    "image_path": s["reference_image"]
                })
        
        # If no image found but we have subjects, just put the path anyway
        if not representative_samples and subjects_data:
            representative_samples.append({
                "asin": subjects_data[0]["asin"],
                "categorical_name": subjects_data[0]["categorical_name"],
                "image_path": subjects_data[0]["reference_image"]
            })

        stats["total_subjects"] += num_subjects
        stats["total_variations"] += num_variations

        stats["categories"][cat_dir] = {
            "clean_name": clean_category_name(cat_dir),
            "num_subjects": num_subjects,
            "num_variations": num_variations,
            "avg_variations_per_subject": round(num_variations / num_subjects, 2) if num_subjects > 0 else 0,
            "categorical_names_preview": sorted(list(categorical_names))[:5],
            "representative_samples": representative_samples,
            "subjects_list": subjects_data
        }

    return stats

def get_html_template(stats_json_str: str) -> str:
    """Returns the HTML report template containing the dashboard code."""
    # We escape double braces for python formatting, i.e. { -> {{ and } -> }}
    # Also we use $ for template variable injection dynamically or python string formatting.
    # To keep it extremely simple, we will replace $${ with template variables inside Python code or format it manually.
    # Wait, let's write the template as raw string and insert stats_json_str.
    # So we don't have to escape anything except double braces if we use .format() or f-string.
    # To avoid formatting complexity, let's just do standard Python string concatenation or .replace() for injecting the JSON data!
    # That is much cleaner, safer, and prevents any brace escaping bugs!
    html_raw = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Statistics Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-card: #1f2937;
            --border-color: #374151;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --accent: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --font-display: 'Outfit', 'Inter', sans-serif;
            --font-sans: 'Inter', sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-sans);
            padding: 2.5rem;
            min-height: 100vh;
            line-height: 1.5;
        }

        header {
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title h1 {
            font-family: var(--font-display);
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        .header-title p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .badge-dir {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        /* Metrics Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .metric-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .metric-card:hover {
            transform: translateY(-4px);
            border-color: var(--primary);
            box-shadow: 0 10px 15px -3px var(--primary-glow), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, var(--primary), var(--accent));
        }

        .metric-label {
            font-family: var(--font-display);
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-family: var(--font-display);
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Content Sections */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2.5rem;
        }

        @media (min-width: 1024px) {
            .main-layout {
                grid-template-columns: 1fr 1fr;
            }
            .full-width-section {
                grid-column: span 2;
            }
        }

        .section-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.75rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .section-card h2 {
            font-family: var(--font-display);
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
        }

        /* Table Styling */
        .table-container {
            overflow-x: auto;
            max-height: 480px;
            overflow-y: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }

        th {
            background-color: var(--bg-card);
            color: var(--text-primary);
            font-family: var(--font-display);
            font-weight: 600;
            padding: 0.75rem 1rem;
            position: sticky;
            top: 0;
            z-index: 10;
            border-bottom: 2px solid var(--border-color);
        }

        td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
        }

        tr:hover td {
            background-color: rgba(255, 255, 255, 0.02);
            color: var(--text-primary);
        }

        .clickable-row {
            cursor: pointer;
        }

        /* Gallery Grid */
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .gallery-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .gallery-card:hover {
            transform: translateY(-6px);
            border-color: var(--accent);
            box-shadow: 0 12px 20px -5px rgba(139, 92, 246, 0.25);
        }

        .gallery-img-container {
            width: 100%;
            height: 200px;
            background-color: var(--bg-primary);
            overflow: hidden;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid var(--border-color);
        }

        .gallery-img-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }

        .gallery-card:hover .gallery-img-container img {
            transform: scale(1.06);
        }

        .gallery-info {
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }

        .gallery-title {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .gallery-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
            margin-top: auto;
        }

        .stat-pill {
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }

        .gallery-preview-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            background-color: rgba(99, 102, 241, 0.08);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            border: 1px dashed rgba(99, 102, 241, 0.3);
        }

        /* Search & Filter Bar */
        .controls-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .search-input {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 0.6rem 1rem;
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 0.9rem;
            width: 100%;
            max-width: 320px;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        /* Modal styling */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(8px);
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
        }

        .modal-content {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 1.5rem;
            max-width: 900px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            animation: modalFadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes modalFadeIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        .modal-header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h3 {
            font-family: var(--font-display);
            font-size: 1.5rem;
            font-weight: 700;
        }

        .close-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 2rem;
            cursor: pointer;
            transition: color 0.2s;
            line-height: 1;
        }

        .close-btn:hover {
            color: var(--text-primary);
        }

        .modal-body {
            padding: 2rem;
        }

        .modal-info-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        @media (min-width: 768px) {
            .modal-info-grid {
                grid-template-columns: 350px 1fr;
            }
        }

        .modal-img-container {
            width: 100%;
            height: 350px;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-img-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .modal-details h4 {
            font-family: var(--font-display);
            font-size: 1.1rem;
            margin-bottom: 0.75rem;
            color: var(--text-primary);
            border-bottom: 1px dashed var(--border-color);
            padding-bottom: 0.25rem;
        }

        .subjects-scroller {
            max-height: 350px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            background-color: var(--bg-primary);
        }

        .subject-item {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            transition: background-color 0.2s;
        }

        .subject-item:last-child {
            border-bottom: none;
        }

        .subject-item:hover {
            background-color: var(--bg-card);
        }

        .subject-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .subject-asin {
            font-family: monospace;
            font-weight: 700;
            color: var(--primary);
        }

        .subject-title {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .subject-prompt-box {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .subject-prompt-box strong {
            color: var(--text-primary);
            display: block;
            margin-bottom: 0.15rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Dataset Statistics Dashboard</h1>
            <p>Analysis and Visual Overview of the dataset categories</p>
        </div>
        <div class="badge-dir" id="dir-badge">/root/Desktop/workspace/woosung/AMZN-review-2023/...</div>
    </header>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Categories</div>
            <div class="metric-value" id="val-categories">0</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Subjects (ASINs)</div>
            <div class="metric-value" id="val-subjects">0</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Variations</div>
            <div class="metric-value" id="val-variations">0</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Variations / Subject</div>
            <div class="metric-value" id="val-avg-variations">0</div>
        </div>
    </div>

    <div class="main-layout">
        <!-- Chart Section -->
        <div class="section-card">
            <h2>Subjects & Variations distribution</h2>
            <div style="position: relative; height: 380px; width: 100%;">
                <canvas id="distribution-chart"></canvas>
            </div>
        </div>

        <!-- Table Section -->
        <div class="section-card">
            <h2>Category statistics</h2>
            <div class="controls-row">
                <input type="text" class="search-input" id="search-box" placeholder="Search categories..." oninput="filterTable()">
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Subjects</th>
                            <th>Variations</th>
                            <th>Avg. Variations</th>
                        </tr>
                    </thead>
                    <tbody id="stats-table-body">
                        <!-- Filled by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Image Overview Section -->
        <div class="section-card full-width-section">
            <h2>At a glance: Category preview gallery</h2>
            <div class="gallery-grid" id="gallery-grid-container">
                <!-- Filled by JS -->
            </div>
        </div>
    </div>

    <!-- Details Modal -->
    <div id="details-modal" class="modal" onclick="closeModalOnOutsideClick(event)">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-category-name">Category Details</h3>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-info-grid">
                    <div>
                        <div class="modal-img-container">
                            <img id="modal-image" src="" alt="Category Preview">
                        </div>
                        <div style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.85rem; text-align: center;" id="modal-image-caption">
                            Preview Image
                        </div>
                    </div>
                    <div class="modal-details">
                        <h4 style="margin-top: 0;">Subjects Details</h4>
                        <div class="subjects-scroller" id="modal-subjects-list">
                            <!-- Filled by JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Inject statistics JSON here
        const data = __INJECTED_STATS_DATA__;

        // Initialize UI Elements
        document.getElementById('dir-badge').innerText = data.dataset_directory;
        document.getElementById('val-categories').innerText = data.total_categories;
        document.getElementById('val-subjects').innerText = data.total_subjects;
        document.getElementById('val-variations').innerText = data.total_variations;
        
        const overallAvg = data.total_subjects > 0 ? (data.total_variations / data.total_subjects).toFixed(2) : 0;
        document.getElementById('val-avg-variations').innerText = overallAvg;

        // Build charts
        const categories = Object.keys(data.categories);
        const subjectsCounts = categories.map(cat => data.categories[cat].num_subjects);
        const variationsCounts = categories.map(cat => data.categories[cat].num_variations);
        const cleanNames = categories.map(cat => data.categories[cat].clean_name);

        const ctx = document.getElementById('distribution-chart').getContext('2d');
        const distributionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: cleanNames,
                datasets: [
                    {
                        label: 'Subjects',
                        data: subjectsCounts,
                        backgroundColor: '#6366f1',
                        borderRadius: 4
                    },
                    {
                        label: 'Variations',
                        data: variationsCounts,
                        backgroundColor: '#8b5cf6',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: false,
                        grid: { display: false },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Inter', size: 9 },
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        grid: { color: '#374151' },
                        ticks: {
                            color: '#9ca3af',
                            font: { family: 'Inter' }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#f3f4f6',
                            font: { family: 'Outfit', weight: '600' }
                        }
                    }
                }
            }
        });

        // Build Table
        const tableBody = document.getElementById('stats-table-body');
        function buildTable() {
            tableBody.innerHTML = '';
            categories.forEach(catKey => {
                const cat = data.categories[catKey];
                const tr = document.createElement('tr');
                tr.className = 'clickable-row';
                tr.onclick = () => openCategoryDetails(catKey);
                tr.innerHTML = `
                    <td style="font-weight:600; color: var(--text-primary); font-family: var(--font-display);">${cat.clean_name}</td>
                    <td>${cat.num_subjects}</td>
                    <td>${cat.num_variations}</td>
                    <td>${cat.avg_variations_per_subject}</td>
                `;
                tableBody.appendChild(tr);
            });
        }
        buildTable();

        function filterTable() {
            const filter = document.getElementById('search-box').value.toLowerCase();
            const rows = tableBody.getElementsByTagName('tr');
            categories.forEach((catKey, index) => {
                const cat = data.categories[catKey];
                if (cat.clean_name.toLowerCase().includes(filter) || catKey.toLowerCase().includes(filter)) {
                    rows[index].style.display = '';
                } else {
                    rows[index].style.display = 'none';
                }
            });
        }

        // Build Gallery
        const galleryGrid = document.getElementById('gallery-grid-container');
        categories.forEach(catKey => {
            const cat = data.categories[catKey];
            const sample = cat.representative_samples && cat.representative_samples.length > 0 ? cat.representative_samples[0] : null;
            const imgPath = sample ? sample.image_path : '';
            
            const card = document.createElement('div');
            card.className = 'gallery-card';
            card.onclick = () => openCategoryDetails(catKey);

            let imgHtml = '<div class="gallery-img-container"><div style="color:var(--text-secondary);font-size:0.8rem;">No Preview Image</div></div>';
            if (imgPath) {
                imgHtml = `
                    <div class="gallery-img-container">
                        <img src="${imgPath}" alt="${cat.clean_name}" onerror="this.style.display='none'; this.parentNode.innerHTML='<div style=\\'color:var(--text-secondary);font-size:0.8rem;\\'>Image Missing</div>';">
                    </div>
                `;
            }

            const categoricalNamePreview = cat.categorical_names_preview && cat.categorical_names_preview.length > 0 
                ? cat.categorical_names_preview.join(', ') 
                : 'No details';

            card.innerHTML = `
                ${imgHtml}
                <div class="gallery-info">
                    <div class="gallery-title">${cat.clean_name}</div>
                    <div class="gallery-preview-label" title="${categoricalNamePreview}">${categoricalNamePreview}</div>
                    <div class="gallery-stats">
                        <span class="stat-pill">Subjects: <strong>${cat.num_subjects}</strong></span>
                        <span class="stat-pill">Variations: <strong>${cat.num_variations}</strong></span>
                    </div>
                </div>
            `;
            galleryGrid.appendChild(card);
        });

        // Modal Functionality
        const modal = document.getElementById('details-modal');
        function openCategoryDetails(catKey) {
            const cat = data.categories[catKey];
            document.getElementById('modal-category-name').innerText = cat.clean_name;
            
            const sample = cat.representative_samples && cat.representative_samples.length > 0 ? cat.representative_samples[0] : null;
            const modalImg = document.getElementById('modal-image');
            const imgCaption = document.getElementById('modal-image-caption');
            
            if (sample) {
                modalImg.src = sample.image_path;
                modalImg.style.display = 'block';
                imgCaption.innerText = `Preview Image (ASIN: ${sample.asin} | ${sample.categorical_name})`;
            } else {
                modalImg.src = '';
                modalImg.style.display = 'none';
                imgCaption.innerText = 'No preview image available';
            }

            // Fill subjects scroller
            const listContainer = document.getElementById('modal-subjects-list');
            listContainer.innerHTML = '';
            
            cat.subjects_list.forEach(sub => {
                const item = document.createElement('div');
                item.className = 'subject-item';
                
                let promptsHtml = '';
                if (sub.variation_prompts && sub.variation_prompts.length > 0) {
                    promptsHtml = sub.variation_prompts.map((p, i) => `
                        <div class="subject-prompt-box">
                            <strong>Variation ${i + 1} Prompt:</strong>
                            ${p}
                        </div>
                    `).join('');
                }

                item.innerHTML = `
                    <div class="subject-header-row">
                        <span class="subject-asin">${sub.asin}</span>
                        <span class="stat-pill" style="font-size:0.7rem;">${sub.num_variations} variations</span>
                    </div>
                    <div class="subject-title">${sub.title || 'Untitled Subject'}</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom: 0.25rem;">
                        <strong>Categorical Name:</strong> ${sub.categorical_name || 'N/A'}
                    </div>
                    ${promptsHtml}
                `;
                listContainer.appendChild(item);
            });

            modal.style.display = 'flex';
        }

        document.addEventListener('keydown', function(event) {
            if (event.key === "Escape") {
                closeModal();
            }
        });

        function closeModal() {
            modal.style.display = 'none';
        }

        function closeModalOnOutsideClick(event) {
            if (event.target === modal) {
                closeModal();
            }
        }
    </script>
</body>
</html>
"""
    return html_raw.replace("__INJECTED_STATS_DATA__", stats_json_str)

def main():
    args = parse_args()
    
    # Resolve output paths
    dataset_dir = args.dataset_dir
    
    if args.output_json is None:
        output_json = os.path.join(dataset_dir, "dataset_stats.json")
    else:
        output_json = args.output_json
        
    if args.output_html is None:
        output_html = os.path.join(dataset_dir, "dataset_stats.html")
    else:
        output_html = args.output_html

    logger.info(f"Scanning dataset in: {dataset_dir}")
    stats = generate_statistics(dataset_dir)

    # Save JSON report
    logger.info(f"Saving JSON statistics to: {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    # Save HTML report
    # We must dump the stats into the template. Let's serialize the stats object to JSON
    # we don't need subjects_list inside the JSON serialized object that goes to HTML
    # actually, keeping the subjects list in HTML makes it fully self-contained and allows
    # displaying details in the modal! It's small enough (30 categories with ~10 items each is only ~300 items).
    stats_json_str = json.dumps(stats, ensure_ascii=False)
    
    html_content = get_html_template(stats_json_str)
    
    logger.info(f"Saving HTML visualization to: {output_html}")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Success! Statistics and visualization generated:")
    print(f"JSON: {output_json}")
    print(f"HTML: {output_html}")

if __name__ == "__main__":
    main()
