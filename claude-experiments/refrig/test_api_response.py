#!/usr/bin/env python3
"""Test script to see what the SerpApi response looks like"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
SERP_API_KEY = os.getenv('SERP_API_TOKEN')

# Test one product
params = {
    'engine': 'home_depot_product',
    'product_id': '325158253',
    'delivery_zip': '46203',
    'api_key': SERP_API_KEY
}

response = requests.get('https://serpapi.com/search.json', params=params)
data = response.json()

# Save full response to see structure
with open('api_response_sample.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Full API response saved to api_response_sample.json")
print("\nKeys in response:")
print(json.dumps(list(data.keys()), indent=2))

if 'product_results' in data:
    print("\nKeys in product_results:")
    print(json.dumps(list(data['product_results'].keys()), indent=2))
