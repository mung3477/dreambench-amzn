import os
import json
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordy")

def get_category_dirs():
    """Returns a list of directory names starting with 'raw_meta_'"""
    dirs = []
    for name in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, name)
        if os.path.isdir(full_path) and name.startswith('raw_meta_'):
            dirs.append(name)
    return sorted(dirs)

def get_metadata_path(category_dir):
    """Returns the path to metadata.json for a category"""
    return os.path.join(BASE_DIR, category_dir, 'metadata.json')

def load_metadata(category_dir):
    """Loads metadata.json for a category, returning an empty list if not found or invalid"""
    path = get_metadata_path(category_dir)
    if not os.path.exists(path):
        logger.warning(f"Metadata file not found: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata from {path}: {e}")
        return []

def save_metadata(category_dir, metadata):
    """Saves metadata to metadata.json for a category with indentation"""
    path = get_metadata_path(category_dir)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully updated metadata: {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving metadata to {path}: {e}")
        return False

# Web Server Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<category>/<path:filename>')
def serve_image(category, filename):
    """Serves images from the specific category directory"""
    category_dir = os.path.join(BASE_DIR, category)
    # Ensure secure directory serving
    return send_from_directory(category_dir, filename)

@app.route('/api/categories', methods=['GET'])
def api_categories():
    """Returns all categories with their counts of remaining variant pairs"""
    categories = []
    category_dirs = get_category_dirs()

    for cat_dir in category_dirs:
        metadata = load_metadata(cat_dir)
        pair_count = 0
        for entry in metadata:
            if 'variation_files' in entry:
                pair_count += len(entry['variation_files'])

        # Make a friendly display name (e.g. raw_meta_All_Beauty -> All Beauty)
        display_name = cat_dir.replace('raw_meta_', '').replace('_', ' ')

        categories.append({
            'id': cat_dir,
            'name': display_name,
            'count': pair_count
        })

    return jsonify(categories)

@app.route('/api/category/<category>', methods=['GET'])
def api_category_detail(category):
    """Returns all individual reference-variant pairs for a category"""
    category_dirs = get_category_dirs()
    if category not in category_dirs:
        return jsonify({'error': 'Category not found'}), 404

    metadata = load_metadata(category)
    pairs = []

    for entry in metadata:
        asin = entry.get('asin')
        title = entry.get('title')
        reference_file = entry.get('reference_file')
        categorical_name = entry.get('categorical_name', '')

        if not asin or not reference_file:
            continue

        for variation in entry.get('variation_files', []):
            pairs.append({
                'asin': asin,
                'title': title,
                'reference_file': reference_file,
                'variation_file': variation.get('file'),
                'prompt': variation.get('prompt'),
                'categorical_name': categorical_name
            })

    return jsonify({'pairs': pairs})

@app.route('/api/delete-variant', methods=['POST'])
def api_delete_variant():
    """Deletes a specific variant image file and its metadata"""
    data = request.json
    category = data.get('category')
    asin = data.get('asin')
    variation_file = data.get('variation_file')

    if not category or not asin or not variation_file:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    category_dirs = get_category_dirs()
    if category not in category_dirs:
        return jsonify({'success': False, 'error': 'Category not found'}), 404

    metadata = load_metadata(category)
    updated = False

    # 1. Update metadata
    for entry in metadata:
        if entry.get('asin') == asin:
            original_len = len(entry.get('variation_files', []))
            entry['variation_files'] = [v for v in entry.get('variation_files', []) if v.get('file') != variation_file]
            if len(entry['variation_files']) < original_len:
                updated = True
            break

    if not updated:
        return jsonify({'success': False, 'error': 'Variant metadata not found'}), 404

    # Save metadata first to ensure sync
    if not save_metadata(category, metadata):
        return jsonify({'success': False, 'error': 'Failed to save metadata'}), 500

    # 2. Delete file from disk
    file_path = os.path.join(BASE_DIR, category, variation_file)
    deleted_from_disk = False
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted variant file: {file_path}")
            deleted_from_disk = True
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return jsonify({'success': False, 'error': f"Failed to delete file from disk: {e}"}), 500
    else:
        logger.warning(f"Variant file not found on disk: {file_path}")
        deleted_from_disk = True # Count as success since file is not there

    return jsonify({'success': True, 'file_deleted': deleted_from_disk})

@app.route('/api/delete-ref-set', methods=['POST'])
def api_delete_ref_set():
    """Deletes the reference image, all variant images, and the metadata entry for the ASIN"""
    data = request.json
    category = data.get('category')
    asin = data.get('asin')

    if not category or not asin:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    category_dirs = get_category_dirs()
    if category not in category_dirs:
        return jsonify({'success': False, 'error': 'Category not found'}), 404

    metadata = load_metadata(category)
    product_entry = None

    # Find entry
    for entry in metadata:
        if entry.get('asin') == asin:
            product_entry = entry
            break

    if not product_entry:
        return jsonify({'success': False, 'error': 'Product entry not found in metadata'}), 404

    # Remove entry from metadata list
    metadata = [entry for entry in metadata if entry.get('asin') != asin]

    # Save updated metadata
    if not save_metadata(category, metadata):
        return jsonify({'success': False, 'error': 'Failed to save metadata'}), 500

    # Delete reference image
    ref_file = product_entry.get('reference_file')
    if ref_file:
        ref_path = os.path.join(BASE_DIR, category, ref_file)
        if os.path.exists(ref_path):
            try:
                os.remove(ref_path)
                logger.info(f"Deleted reference file: {ref_path}")
            except Exception as e:
                logger.error(f"Error deleting reference file {ref_path}: {e}")
        else:
            logger.warning(f"Reference file not found on disk: {ref_path}")

    # Delete all variant images
    for variation in product_entry.get('variation_files', []):
        var_file = variation.get('file')
        if var_file:
            var_path = os.path.join(BASE_DIR, category, var_file)
            if os.path.exists(var_path):
                try:
                    os.remove(var_path)
                    logger.info(f"Deleted variant file: {var_path}")
                except Exception as e:
                    logger.error(f"Error deleting variant file {var_path}: {e}")
            else:
                logger.warning(f"Variant file not found on disk: {var_path}")

    # Also clean up the ASIN subdirectory if it's empty
    asin_dir = os.path.join(BASE_DIR, category, asin)
    if os.path.exists(asin_dir) and os.path.isdir(asin_dir):
        try:
            # Check if directory is empty or contains only non-essential files
            files_remaining = os.listdir(asin_dir)
            if len(files_remaining) == 0:
                os.rmdir(asin_dir)
                logger.info(f"Deleted empty product directory: {asin_dir}")
        except Exception as e:
            logger.warning(f"Failed to check/remove empty product directory {asin_dir}: {e}")

    return jsonify({'success': True})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Start Dataset Curator & Visualizer Web App")
    parser.add_argument('--port', type=int, default=5000, help="Port to run web app on")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host to run web app on")
    args = parser.parse_args()

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
