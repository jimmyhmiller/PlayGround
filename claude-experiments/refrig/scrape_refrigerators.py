#!/usr/bin/env python3
"""
Scrape refrigerator details from Home Depot and Lowe's URLs.
Saves results to refrigerator_comparison.json
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urlparse

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


def get_headers():
    """Return headers to mimic a real browser"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }


def extract_homedepot_info(soup, url):
    """Extract refrigerator info from Home Depot page"""
    info = {
        'url': url,
        'store': 'Home Depot',
        'model': None,
        'brand': None,
        'capacity': None,
        'dimensions': {},
        'price': None,
        'features': [],
        'images': []
    }

    # Try to get product title
    title_elem = soup.find('h1', class_='product-details__title')
    if title_elem:
        info['title'] = title_elem.get_text(strip=True)
        # Extract brand from title
        brand_match = re.match(r'^(\w+)', title_elem.get_text(strip=True))
        if brand_match:
            info['brand'] = brand_match.group(1)

    # Try to find model number
    model_elem = soup.find('div', string=re.compile(r'Model #', re.I))
    if model_elem:
        info['model'] = model_elem.find_next_sibling().get_text(strip=True) if model_elem.find_next_sibling() else None

    # Look for specifications table
    spec_tables = soup.find_all('table', class_='specifications__table')
    for table in spec_tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['th', 'td'])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)

                if 'height' in key:
                    info['dimensions']['height'] = value
                elif 'width' in key:
                    info['dimensions']['width'] = value
                elif 'depth' in key:
                    info['dimensions']['depth'] = value
                elif 'capacity' in key or 'cu' in key:
                    info['capacity'] = value

    # Try to get price
    price_elem = soup.find('div', {'data-testid': 'product-price'})
    if not price_elem:
        price_elem = soup.find('span', class_=re.compile(r'price'))
    if price_elem:
        price_text = price_elem.get_text(strip=True)
        price_match = re.search(r'\$[\d,]+\.?\d*', price_text)
        if price_match:
            info['price'] = price_match.group(0)

    # Get images
    img_elements = soup.find_all('img', class_=re.compile(r'product|media'))
    for img in img_elements[:5]:  # Limit to first 5 images
        if img.get('src') and 'homedepot' in img['src']:
            info['images'].append(img['src'])

    return info


def extract_lowes_info(soup, url):
    """Extract refrigerator info from Lowe's page"""
    info = {
        'url': url,
        'store': "Lowe's",
        'model': None,
        'brand': None,
        'capacity': None,
        'dimensions': {},
        'price': None,
        'features': [],
        'images': []
    }

    # Try to get product title
    title_elem = soup.find('h1', class_=re.compile(r'heading|title'))
    if title_elem:
        info['title'] = title_elem.get_text(strip=True)
        # Extract brand from title
        brand_match = re.match(r'^(\w+)', title_elem.get_text(strip=True))
        if brand_match:
            info['brand'] = brand_match.group(1)

    # Look for model number
    model_elem = soup.find(string=re.compile(r'Model #?:', re.I))
    if model_elem:
        # Try to find the model number near this element
        parent = model_elem.find_parent()
        if parent:
            model_text = parent.get_text(strip=True)
            model_match = re.search(r'Model #?:?\s*([A-Z0-9-]+)', model_text, re.I)
            if model_match:
                info['model'] = model_match.group(1)

    # Look for specifications
    spec_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'spec|dimension'))
    for elem in spec_elements:
        text = elem.get_text(strip=True).lower()

        if 'height' in text:
            height_match = re.search(r'(\d+\.?\d*)\s*(?:in|inches)', text, re.I)
            if height_match:
                info['dimensions']['height'] = height_match.group(0)
        elif 'width' in text:
            width_match = re.search(r'(\d+\.?\d*)\s*(?:in|inches)', text, re.I)
            if width_match:
                info['dimensions']['width'] = width_match.group(0)
        elif 'depth' in text:
            depth_match = re.search(r'(\d+\.?\d*)\s*(?:in|inches)', text, re.I)
            if depth_match:
                info['dimensions']['depth'] = depth_match.group(0)
        elif 'capacity' in text or 'cu' in text:
            capacity_match = re.search(r'(\d+\.?\d*)\s*cu\.?\s*ft', text, re.I)
            if capacity_match:
                info['capacity'] = capacity_match.group(0)

    # Try to get price
    price_elem = soup.find('div', class_=re.compile(r'price'))
    if not price_elem:
        price_elem = soup.find('span', {'data-testid': 'price'})
    if price_elem:
        price_text = price_elem.get_text(strip=True)
        price_match = re.search(r'\$[\d,]+\.?\d*', price_text)
        if price_match:
            info['price'] = price_match.group(0)

    # Get images
    img_elements = soup.find_all('img', class_=re.compile(r'product|media'))
    for img in img_elements[:5]:  # Limit to first 5 images
        if img.get('src') and ('lowes' in img['src'] or img['src'].startswith('http')):
            info['images'].append(img['src'])

    return info


def scrape_refrigerator(url):
    """Scrape a single refrigerator page"""
    print(f"Scraping: {url}")

    try:
        response = requests.get(url, headers=get_headers(), timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Determine which store and extract accordingly
        if 'homedepot.com' in url:
            info = extract_homedepot_info(soup, url)
        elif 'lowes.com' in url:
            info = extract_lowes_info(soup, url)
        else:
            info = {'url': url, 'error': 'Unknown store'}

        print(f"  ✓ Successfully scraped {info.get('store', 'unknown')}")
        return info

    except requests.RequestException as e:
        print(f"  ✗ Error scraping {url}: {e}")
        return {'url': url, 'error': str(e)}


def main():
    """Main function to scrape all refrigerators"""
    print("Starting refrigerator scraping...\n")

    results = []

    for url in URLS:
        result = scrape_refrigerator(url)
        results.append(result)
        time.sleep(2)  # Be polite, wait between requests

    # Save results to JSON file
    output_file = 'refrigerator_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Scraping complete! Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i}. ERROR: {result.get('store', 'Unknown')} - {result.get('error', 'Unknown error')}")
        else:
            title = result.get('title', 'Unknown')
            price = result.get('price', 'N/A')
            capacity = result.get('capacity', 'N/A')
            print(f"{i}. {result.get('brand', 'Unknown')} - {capacity} - {price}")

    print(f"\nFor detailed comparison, check {output_file}")


if __name__ == '__main__':
    main()
