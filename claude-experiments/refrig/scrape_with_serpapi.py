#!/usr/bin/env python3
"""
Scrape refrigerator details using SerpApi
Reads API key from .env file
"""

import requests
import json
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

SERP_API_KEY = os.getenv('SERP_API_TOKEN')

# List of refrigerator URLs
URLS = [
    "https://www.homedepot.com/p/LG-28-cu-ft-3-Door-French-Door-Refrigerator-with-Ice-and-Water-Dispenser-and-Craft-Ice-in-PrintProof-Stainless-Steel-LHFS28XBS/325158253",
    "https://www.homedepot.com/p/Samsung-26-cu-ft-Mega-Capacity-Counter-Depth-3-Door-French-Door-Refrigerator-in-Stainless-Steel-with-Four-Types-of-Ice-RF27CG5400SR/326195005",
    "https://www.homedepot.com/p/LG-28-cu-ft-3-Door-French-Door-Refrigerator-with-Ice-and-Water-with-Single-Ice-in-Stainless-Standard-Depth-LRFS28XBS/325158258",
    "https://www.lowes.com/pd/Samsung-26-5-cu-ft-Counter-depth-Built-In-Smart-French-Door-Refrigerator-with-Dual-Ice-Maker-Stainless-Steel-ENERGY-STAR/5014610045",
    "https://www.lowes.com/pd/Whirlpool-Whirlpool-36-Inch-Wide-French-Door-Refrigerator-with-Dual-Ice-Makers-23-cu-ft/5016612147",
    "https://www.lowes.com/pd/Frigidaire-22-6-cu-ft-Counter-depth-French-Door-Refrigerator-with-Ice-Maker-Stainless-Steel-ENERGY-STAR/5013990507",
    "https://www.lowes.com/pd/Samsung-Counter-depth-Mega-Capacity-25-5-cu-ft-Smart-French-Door-Refrigerator-with-Dual-Ice-Maker-Water-and-Ice-Dispenser-Fingerprint-Resistant-Stainless-Steel-ENERGY-STAR/5015371771",
]

DELIVERY_ZIP = "46203"


def extract_product_id(url):
    """Extract product ID from URL"""
    if 'homedepot.com' in url:
        # Home Depot format: /p/.../.../PRODUCT_ID
        match = re.search(r'/p/[^/]+/(\d+)', url)
        return match.group(1) if match else None
    elif 'lowes.com' in url:
        # Lowe's format: /pd/.../PRODUCT_ID
        match = re.search(r'/pd/[^/]+/(\d+)', url)
        return match.group(1) if match else None
    return None


def get_store_type(url):
    """Determine which store the URL is from"""
    if 'homedepot.com' in url:
        return 'home_depot', 'Home Depot'
    elif 'lowes.com' in url:
        return 'lowes', "Lowe's"
    return None, 'Unknown'


def fetch_homedepot_product(product_id):
    """Fetch product data from Home Depot using SerpApi"""
    params = {
        'engine': 'home_depot_product',
        'product_id': product_id,
        'delivery_zip': DELIVERY_ZIP,
        'api_key': SERP_API_KEY
    }

    try:
        response = requests.get('https://serpapi.com/search.json', params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ✗ Error fetching Home Depot product {product_id}: {e}")
        return None


def fetch_lowes_product(product_id):
    """Fetch product data from Lowe's using SerpApi"""
    params = {
        'engine': 'lowes_product',
        'product_id': product_id,
        'api_key': SERP_API_KEY
    }

    try:
        response = requests.get('https://serpapi.com/search.json', params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ✗ Error fetching Lowe's product {product_id}: {e}")
        return None


def parse_homedepot_data(data, url):
    """Parse Home Depot API response into our format"""
    if not data or 'product_results' not in data:
        return {'url': url, 'error': 'No product data found'}

    product = data['product_results']

    # Extract brand - it might be a dict or string
    brand_data = product.get('brand', '')
    if isinstance(brand_data, dict):
        brand = brand_data.get('name', '')
    else:
        brand = brand_data

    info = {
        'url': url,
        'store': 'Home Depot',
        'title': product.get('title', ''),
        'model': product.get('model_number', ''),
        'brand': brand,
        'price': None,
        'capacity': None,
        'dimensions': {},
        'features': [],
        'images': [],
        'specifications': {}
    }

    # Price - direct from product
    if 'price' in product:
        info['price'] = f"${product['price']:.2f}"

    # Images - get first image from each set
    if 'images' in product:
        first_images = []
        for img_set in product['images'][:5]:
            if isinstance(img_set, list) and len(img_set) > 0:
                first_images.append(img_set[0])  # Get first (smallest) image
        info['images'] = first_images

    # Specifications
    if 'specifications' in product:
        specs = product['specifications']
        for section in specs:
            if 'value' in section and isinstance(section['value'], list):
                for spec in section['value']:
                    spec_name = spec.get('name', '').lower()
                    spec_value = spec.get('value', '')

                    info['specifications'][spec.get('name', '')] = spec_value

                    # Extract specific dimensions
                    if spec_name == 'product width (in.)':
                        info['dimensions']['width'] = spec_value
                    elif spec_name == 'product height (in.)':
                        info['dimensions']['height'] = spec_value
                    elif spec_name == 'product depth (in.)':
                        info['dimensions']['depth'] = spec_value
                    elif 'total capacity' in spec_name:
                        info['capacity'] = spec_value

    # Features
    if 'highlights' in product:
        info['features'] = product['highlights']
    elif 'bullets' in product:
        # Clean HTML from bullets
        import re
        bullets = []
        for bullet in product['bullets'][:5]:
            # Remove HTML tags
            clean_bullet = re.sub(r'<[^>]+>', '', bullet)
            if len(clean_bullet) > 10:
                bullets.append(clean_bullet)
        info['features'] = bullets

    return info


def parse_lowes_data(data, url):
    """Parse Lowe's API response into our format"""
    if not data or 'product_results' not in data:
        return {'url': url, 'error': 'No product data found'}

    product = data['product_results']

    info = {
        'url': url,
        'store': "Lowe's",
        'title': product.get('title', ''),
        'model': product.get('model_number', ''),
        'brand': product.get('brand', ''),
        'price': None,
        'capacity': None,
        'dimensions': {},
        'features': [],
        'images': [],
        'specifications': {}
    }

    # Price
    if 'price' in product:
        info['price'] = product['price']

    # Images
    if 'images' in product:
        info['images'] = product['images'][:5]  # Limit to 5 images

    # Specifications
    if 'specifications' in product:
        specs = product['specifications']
        for spec in specs:
            spec_name = spec.get('name', '').lower()
            spec_value = spec.get('value', '')

            info['specifications'][spec.get('name', '')] = spec_value

            # Extract dimensions
            if 'height' in spec_name:
                info['dimensions']['height'] = spec_value
            elif 'width' in spec_name:
                info['dimensions']['width'] = spec_value
            elif 'depth' in spec_name:
                info['dimensions']['depth'] = spec_value
            elif 'capacity' in spec_name or 'cu' in spec_name.lower():
                info['capacity'] = spec_value

    # Features
    if 'highlights' in product:
        info['features'] = product['highlights']
    elif 'features' in product:
        info['features'] = product['features']

    return info


def scrape_refrigerator(url):
    """Scrape a single refrigerator page using SerpApi"""
    product_id = extract_product_id(url)
    if not product_id:
        print(f"✗ Could not extract product ID from: {url}")
        return {'url': url, 'error': 'Could not extract product ID'}

    store_engine, store_name = get_store_type(url)
    print(f"Fetching {store_name} product {product_id}...")

    if store_engine == 'home_depot':
        data = fetch_homedepot_product(product_id)
        if data:
            result = parse_homedepot_data(data, url)
            print(f"  ✓ {result.get('brand', 'Unknown')} - {result.get('capacity', 'N/A')} - {result.get('price', 'N/A')}")
            return result
    elif store_engine == 'lowes':
        data = fetch_lowes_product(product_id)
        if data:
            result = parse_lowes_data(data, url)
            print(f"  ✓ {result.get('brand', 'Unknown')} - {result.get('capacity', 'N/A')} - {result.get('price', 'N/A')}")
            return result

    print(f"  ✗ Failed to fetch data")
    return {'url': url, 'error': 'Failed to fetch data'}


def main():
    """Main function to scrape all refrigerators"""
    if not SERP_API_KEY:
        print("ERROR: SERP_API_TOKEN not found in .env file")
        print("Please create a .env file with: SERP_API_TOKEN=your_api_key")
        return

    print("="*60)
    print("Refrigerator Comparison Scraper with SerpApi")
    print("="*60)
    print(f"Using API key: {SERP_API_KEY[:10]}...")
    print()

    results = []

    for i, url in enumerate(URLS, 1):
        print(f"\n[{i}/{len(URLS)}] {url}")
        result = scrape_refrigerator(url)
        results.append(result)

    # Save results to JSON file
    output_file = 'refrigerator_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary table
    print("\n" + "="*120)
    print(f"{'#':<4} {'Brand':<12} {'Model':<18} {'Capacity':<12} {'Price':<12} {'Dimensions (W x H x D)':<40}")
    print("="*120)

    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i:<4} ERROR: {result.get('error', 'Unknown error')}")
        else:
            brand = str(result.get('brand', 'Unknown'))[:11]
            model = str(result.get('model', 'N/A'))[:17]
            capacity = str(result.get('capacity', 'N/A'))[:11]
            price = str(result.get('price', 'N/A'))[:11]

            dims = result.get('dimensions', {})
            dim_str = f"{dims.get('width', '?')} x {dims.get('height', '?')} x {dims.get('depth', '?')}"

            print(f"{i:<4} {brand:<12} {model:<18} {capacity:<12} {price:<12} {dim_str:<40}")

    print("="*120)
    print(f"\nFor full details, check {output_file}")


if __name__ == '__main__':
    main()
