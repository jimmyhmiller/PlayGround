#!/usr/bin/env python3
"""
Parse Lowe's refrigerator data from text files
"""

import json
import re

def parse_lowes_file(filename):
    """Parse a single Lowe's refrigerator text file"""
    with open(filename, 'r') as f:
        content = f.read()

    # Extract URL (first line)
    lines = content.strip().split('\n')
    url = lines[0] if lines else ''

    # Initialize data structure
    data = {
        'url': url,
        'store': "Lowe's",
        'title': '',
        'model': '',
        'brand': '',
        'price': None,
        'capacity': None,
        'dimensions': {},
        'features': [],
        'specifications': {}
    }

    # Extract brand from URL
    if 'Samsung' in url:
        data['brand'] = 'Samsung'
    elif 'Whirlpool' in url:
        data['brand'] = 'Whirlpool'
    elif 'Frigidaire' in url:
        data['brand'] = 'Frigidaire'
    elif 'LG' in url:
        data['brand'] = 'LG'

    # Parse key-value pairs
    current_key = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for dimension keys
        if 'Actual Width (Inches)' in line:
            current_key = 'width'
        elif 'Product Height (in.)' in line or 'Height to Top of Door Hinge (Inches)' in line:
            current_key = 'height'
        elif 'Depth (Including Handles) (Inches)' in line:
            current_key = 'depth'
        elif 'Overall Capacity (Cu. Feet)' in line:
            current_key = 'capacity'
        elif 'Refrigerator Capacity (Cu. Feet)' in line:
            current_key = 'ref_capacity'
        elif 'Freezer Capacity (Cu. Feet)' in line:
            current_key = 'freezer_capacity'
        elif 'Weight (lbs.)' in line:
            current_key = 'weight'
        elif 'Depth Type' in line:
            current_key = 'depth_type'
        elif 'Ice Maker' in line and current_key != 'ice_maker':
            current_key = 'ice_maker'
        elif 'Smart Compatible' in line:
            current_key = 'smart'
        elif 'ENERGY STAR Certified' in line:
            current_key = 'energy_star'
        elif 'Series Name' in line:
            current_key = 'series'
        elif 'Manufacturer Color/Finish' in line:
            current_key = 'finish'
        elif 'Ice Type' in line:
            current_key = 'ice_type'
        elif 'Depth (Excluding Handles)' in line:
            current_key = 'depth_no_handles'
        else:
            # Try to match value to current key
            if current_key:
                # Look for numeric values
                if current_key == 'width':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['dimensions']['width'] = f"{match.group(1)} in"
                        current_key = None
                elif current_key == 'height':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['dimensions']['height'] = f"{match.group(1)} in"
                        current_key = None
                elif current_key == 'depth':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['dimensions']['depth'] = f"{match.group(1)} in"
                        current_key = None
                elif current_key == 'depth_no_handles':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['dimensions']['depth_no_handles'] = f"{match.group(1)} in"
                        current_key = None
                elif current_key == 'capacity':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['capacity'] = f"{match.group(1)} cu ft"
                        current_key = None
                elif current_key == 'ref_capacity':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['specifications']['Refrigerator Capacity'] = f"{match.group(1)} cu ft"
                        current_key = None
                elif current_key == 'freezer_capacity':
                    match = re.match(r'^(\d+\.?\d*)$', line)
                    if match:
                        data['specifications']['Freezer Capacity'] = f"{match.group(1)} cu ft"
                        current_key = None
                elif current_key == 'weight':
                    match = re.match(r'^(\d+)$', line)
                    if match:
                        data['specifications']['Weight'] = f"{match.group(1)} lbs"
                        current_key = None
                elif current_key == 'depth_type':
                    if line in ['Counter-Depth', 'Standard-Depth']:
                        data['specifications']['Installation Depth'] = line
                        current_key = None
                elif current_key == 'ice_maker':
                    if line in ['Single', 'Dual', 'Triple']:
                        data['specifications']['Ice Maker'] = line
                        if line == 'Dual':
                            data['features'].append('Dual ice maker')
                        elif line == 'Triple':
                            data['features'].append('Triple ice maker')
                        current_key = None
                elif current_key == 'smart':
                    if line in ['Yes', 'No']:
                        data['specifications']['Smart Compatible'] = line
                        if line == 'Yes':
                            data['features'].append('Smart home enabled')
                        current_key = None
                elif current_key == 'energy_star':
                    if line in ['Yes', 'No']:
                        data['specifications']['Energy Star'] = line
                        if line == 'Yes':
                            data['features'].append('Energy Star certified')
                        current_key = None
                elif current_key == 'series':
                    if line and line not in ['Not Applicable', '']:
                        data['specifications']['Series'] = line
                        current_key = None
                elif current_key == 'finish':
                    if line and 'Stainless' in line:
                        data['specifications']['Finish'] = line
                        if 'Fingerprint' in line:
                            data['features'].append('Fingerprint-resistant finish')
                        current_key = None
                elif current_key == 'ice_type':
                    if line:
                        data['specifications']['Ice Type'] = line
                        current_key = None

    # Generate title from brand and capacity
    if data['brand'] and data['capacity']:
        depth_type = data['specifications'].get('Installation Depth', '')
        data['title'] = f"{data['brand']} {data['capacity']} {depth_type} French Door Refrigerator"

    return data


def main():
    """Parse all Lowe's refrigerator files"""
    files = ['fridge1.txt', 'fridge2.txt', 'fridge3.txt', 'fridge4.txt']

    lowes_data = []

    for filename in files:
        print(f"Parsing {filename}...")
        try:
            data = parse_lowes_file(filename)
            lowes_data.append(data)

            print(f"  ✓ {data['brand']} - {data['capacity']} - {data['dimensions'].get('width', '?')} x {data['dimensions'].get('height', '?')} x {data['dimensions'].get('depth', '?')}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Load existing Home Depot data
    try:
        with open('refrigerator_comparison.json', 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []

    # Filter out old failed Lowe's entries
    home_depot_data = [item for item in existing_data if item.get('store') == 'Home Depot']

    # Combine data
    all_data = home_depot_data + lowes_data

    # Save combined data
    with open('refrigerator_comparison.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Combined data saved to refrigerator_comparison.json")
    print(f"{'='*60}")

    # Print summary table
    print("\n" + "="*120)
    print(f"{'#':<4} {'Store':<12} {'Brand':<12} {'Capacity':<12} {'Type':<18} {'Dimensions (W x H x D)':<35}")
    print("="*120)

    for i, item in enumerate(all_data, 1):
        store = item.get('store', 'Unknown')[:11]
        brand = item.get('brand', 'Unknown')[:11]
        capacity = item.get('capacity', 'N/A')[:11]
        depth_type = item.get('specifications', {}).get('Installation Depth', 'N/A')[:17]

        dims = item.get('dimensions', {})
        dim_str = f"{dims.get('width', '?')} x {dims.get('height', '?')} x {dims.get('depth', '?')}"

        print(f"{i:<4} {store:<12} {brand:<12} {capacity:<12} {depth_type:<18} {dim_str:<35}")

    print("="*120)


if __name__ == '__main__':
    main()
