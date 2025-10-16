#!/usr/bin/env python3
"""
Scrape refrigerator details using Playwright for better JavaScript support.
Run with: playwright install chromium (first time only)
Then: python scrape_with_playwright.py
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import json
import time
import re

URLS = [
    "https://www.homedepot.com/p/LG-28-cu-ft-3-Door-French-Door-Refrigerator-with-Ice-and-Water-Dispenser-and-Craft-Ice-in-PrintProof-Stainless-Steel-LHFS28XBS/325158253",
    "https://www.homedepot.com/p/Samsung-26-cu-ft-Mega-Capacity-Counter-Depth-3-Door-French-Door-Refrigerator-in-Stainless-Steel-with-Four-Types-of-Ice-RF27CG5400SR/326195005",
    "https://www.homedepot.com/p/LG-28-cu-ft-3-Door-French-Door-Refrigerator-with-Ice-and-Water-with-Single-Ice-in-Stainless-Standard-Depth-LRFS28XBS/325158258",
    "https://www.lowes.com/pd/Samsung-26-5-cu-ft-Counter-depth-Built-In-Smart-French-Door-Refrigerator-with-Dual-Ice-Maker-Stainless-Steel-ENERGY-STAR/5014610045",
    "https://www.lowes.com/pd/Whirlpool-Whirlpool-36-Inch-Wide-French-Door-Refrigerator-with-Dual-Ice-Makers-23-cu-ft/5016612147",
    "https://www.lowes.com/pd/Frigidaire-22-6-cu-ft-Counter-depth-French-Door-Refrigerator-with-Ice-Maker-Stainless-Steel-ENERGY-STAR/5013990507",
    "https://www.lowes.com/pd/Samsung-Counter-depth-Mega-Capacity-25-5-cu-ft-Smart-French-Door-Refrigerator-with-Dual-Ice-Maker-Water-and-Ice-Dispenser-Fingerprint-Resistant-Stainless-Steel-ENERGY-STAR/5015371771",
]


def extract_info_from_page(page, url):
    """Extract refrigerator information from the page"""
    info = {
        'url': url,
        'store': 'Home Depot' if 'homedepot' in url else "Lowe's",
        'title': None,
        'model': None,
        'brand': None,
        'capacity': None,
        'dimensions': {},
        'price': None,
        'features': [],
        'images': []
    }

    try:
        # Get page title
        title = page.title()
        info['title'] = title

        # Extract text content
        content = page.content()

        # Try to get price (various selectors for both sites)
        price_selectors = [
            '[data-testid="product-price"]',
            '.price',
            '.price-format__main-price',
            '[data-component="PriceDisplay"]',
            'span[itemprop="price"]',
        ]

        for selector in price_selectors:
            try:
                price_elem = page.locator(selector).first
                if price_elem.is_visible(timeout=2000):
                    price_text = price_elem.text_content()
                    price_match = re.search(r'\$[\d,]+\.?\d*', price_text)
                    if price_match:
                        info['price'] = price_match.group(0)
                        break
            except:
                continue

        # Look for model number in the content
        model_patterns = [
            r'Model[#\s:]+([A-Z0-9-]+)',
            r'Model Number[:\s]+([A-Z0-9-]+)',
            r'Item Model Number[:\s]+([A-Z0-9-]+)',
        ]

        for pattern in model_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                info['model'] = match.group(1)
                break

        # Extract from title
        if info['title']:
            # Try to get brand
            brand_match = re.match(r'^(\w+)', info['title'])
            if brand_match:
                info['brand'] = brand_match.group(1)

            # Try to get capacity
            capacity_match = re.search(r'(\d+\.?\d*)\s*cu\.?\s*ft', info['title'], re.IGNORECASE)
            if capacity_match:
                info['capacity'] = f"{capacity_match.group(1)} cu ft"

        # Look for dimensions in specifications
        spec_patterns = {
            'height': r'Height.*?(\d+\.?\d*)\s*(?:in|inches)',
            'width': r'Width.*?(\d+\.?\d*)\s*(?:in|inches)',
            'depth': r'Depth.*?(\d+\.?\d*)\s*(?:in|inches)',
        }

        for dim_name, pattern in spec_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                info['dimensions'][dim_name] = f"{match.group(1)} inches"

        # Get images
        try:
            images = page.locator('img[src*="product"], img[src*="media"]').all()
            for img in images[:5]:
                src = img.get_attribute('src')
                if src and src.startswith('http'):
                    info['images'].append(src)
        except:
            pass

        return info

    except Exception as e:
        info['error'] = str(e)
        return info


def scrape_all_refrigerators():
    """Main function to scrape all refrigerators"""
    results = []

    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        for i, url in enumerate(URLS, 1):
            print(f"\n[{i}/{len(URLS)}] Scraping: {url}")

            try:
                # Navigate to page
                page.goto(url, wait_until='domcontentloaded', timeout=60000)

                # Wait a bit for dynamic content
                page.wait_for_timeout(3000)

                # Extract information
                info = extract_info_from_page(page, url)
                results.append(info)

                print(f"  ✓ {info.get('brand', 'Unknown')} - {info.get('capacity', 'N/A')} - {info.get('price', 'N/A')}")

                # Be polite
                time.sleep(2)

            except PlaywrightTimeout:
                print(f"  ✗ Timeout loading page")
                results.append({'url': url, 'error': 'Timeout'})
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results.append({'url': url, 'error': str(e)})

        browser.close()

    # Save results
    output_file = 'refrigerator_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("Summary:\n")
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i}. ERROR: {result.get('error', 'Unknown error')}")
        else:
            brand = result.get('brand', 'Unknown')
            capacity = result.get('capacity', 'N/A')
            price = result.get('price', 'N/A')
            dims = result.get('dimensions', {})
            dim_str = f"{dims.get('width', '?')} x {dims.get('height', '?')} x {dims.get('depth', '?')}"
            print(f"{i}. {brand} - {capacity} - {price}")
            print(f"   Dimensions: {dim_str}")


if __name__ == '__main__':
    print("="*60)
    print("Refrigerator Comparison Scraper with Playwright")
    print("="*60)
    scrape_all_refrigerators()
