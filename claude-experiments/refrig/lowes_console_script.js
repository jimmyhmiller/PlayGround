/**
 * LOWE'S REFRIGERATOR DATA EXTRACTOR
 *
 * How to use:
 * 1. Go to a Lowe's refrigerator product page
 * 2. Open browser console (F12 or Right Click > Inspect > Console tab)
 * 3. Copy and paste this entire script
 * 4. Press Enter
 * 5. A popup will appear with the data - click "Copy JSON"
 * 6. Save to a text file
 */

(function() {
    console.log('🧊 Lowe\'s Refrigerator Data Extractor Starting...');

    const data = {
        url: window.location.href,
        store: "Lowe's",
        title: null,
        model: null,
        brand: null,
        price: null,
        capacity: null,
        dimensions: {},
        features: [],
        specifications: {}
    };

    // Extract title
    const titleElement = document.querySelector('h1[itemprop="name"], h1.art-pd-header__title, h1');
    if (titleElement) {
        data.title = titleElement.textContent.trim();
        console.log('✓ Title:', data.title);
    }

    // Extract price
    const priceSelectors = [
        '[data-selector="buybox-price"]',
        '.art-pd-price__dollars',
        '[itemprop="price"]',
        '.price',
        'span[data-testid="price"]'
    ];

    for (const selector of priceSelectors) {
        const priceElem = document.querySelector(selector);
        if (priceElem) {
            const priceText = priceElem.textContent;
            const priceMatch = priceText.match(/\$?[\d,]+\.?\d*/);
            if (priceMatch) {
                data.price = priceMatch[0].startsWith('$') ? priceMatch[0] : '$' + priceMatch[0];
                console.log('✓ Price:', data.price);
                break;
            }
        }
    }

    // Extract brand from title or page
    const brandMatch = data.title ? data.title.match(/^(\w+)\s/) : null;
    if (brandMatch) {
        data.brand = brandMatch[1];
    }
    console.log('✓ Brand:', data.brand);

    // Extract model number
    const modelSelectors = [
        'span[itemprop="model"]',
        'div:contains("Model #")',
        'span.modelNumber'
    ];

    const bodyText = document.body.innerText;
    const modelMatch = bodyText.match(/(?:Model|Item)\s*#?\s*:?\s*([A-Z0-9-]+)/i);
    if (modelMatch) {
        data.model = modelMatch[1];
        console.log('✓ Model:', data.model);
    }

    // Function to find specs in various table formats
    function extractSpecs() {
        // Try to find specifications section
        const specSections = [
            ...document.querySelectorAll('[data-testid="specifications"]'),
            ...document.querySelectorAll('.specifications'),
            ...document.querySelectorAll('[class*="spec"]'),
            ...document.querySelectorAll('table')
        ];

        for (const section of specSections) {
            const rows = section.querySelectorAll('tr, div[class*="row"]');

            for (const row of rows) {
                const text = row.textContent.toLowerCase();
                const fullText = row.textContent.trim();

                // Dimensions
                if (text.includes('actual width') && text.includes('inches')) {
                    const match = fullText.match(/([\d.]+)\s*(?:in|inches)/i);
                    if (match) data.dimensions.width = match[1] + ' in';
                }
                if ((text.includes('height') && text.includes('top') && text.includes('hinge')) ||
                    (text.includes('overall height'))) {
                    const match = fullText.match(/([\d.]+)\s*(?:in|inches)/i);
                    if (match) data.dimensions.height = match[1] + ' in';
                }
                if (text.includes('depth') && text.includes('including handles')) {
                    const match = fullText.match(/([\d.]+)\s*(?:in|inches)/i);
                    if (match) data.dimensions.depth = match[1] + ' in';
                }
                if (text.includes('depth') && text.includes('excluding handles')) {
                    const match = fullText.match(/([\d.]+)\s*(?:in|inches)/i);
                    if (match) data.dimensions.depth_no_handles = match[1] + ' in';
                }

                // Capacity
                if (text.includes('overall capacity') && text.includes('cu')) {
                    const match = fullText.match(/([\d.]+)\s*(?:cu\.?\s*ft)/i);
                    if (match) data.capacity = match[1] + ' cu ft';
                }

                // Other specs
                if (text.includes('depth type')) {
                    if (text.includes('counter')) data.specifications['Installation Depth'] = 'Counter-Depth';
                    else if (text.includes('standard')) data.specifications['Installation Depth'] = 'Standard-Depth';
                }

                if (text.includes('ice maker') && !text.includes('location')) {
                    if (text.includes('dual')) data.specifications['Ice Maker'] = 'Dual';
                    else if (text.includes('triple')) data.specifications['Ice Maker'] = 'Triple';
                    else if (text.includes('single')) data.specifications['Ice Maker'] = 'Single';
                }

                if (text.includes('smart compatible')) {
                    if (text.includes('yes')) {
                        data.specifications['Smart Compatible'] = 'Yes';
                        data.features.push('Smart home enabled');
                    }
                }

                if (text.includes('energy star') && text.includes('certified')) {
                    if (text.includes('yes')) {
                        data.specifications['Energy Star'] = 'Yes';
                        data.features.push('Energy Star certified');
                    }
                }

                if (text.includes('fingerprint-resistant') || text.includes('fingerprint resistant')) {
                    if (text.includes('yes') || text.includes('resistant stainless')) {
                        data.features.push('Fingerprint-resistant finish');
                    }
                }

                if (text.includes('ice type')) {
                    const iceMatch = fullText.match(/Ice Type[:\s]+([\w/]+)/i);
                    if (iceMatch) data.specifications['Ice Type'] = iceMatch[1];
                }

                if (text.includes('weight') && text.includes('lbs')) {
                    const match = fullText.match(/([\d]+)\s*(?:lb|lbs)/i);
                    if (match) data.specifications['Weight'] = match[1] + ' lbs';
                }
            }
        }
    }

    extractSpecs();

    console.log('✓ Dimensions:', data.dimensions);
    console.log('✓ Capacity:', data.capacity);
    console.log('✓ Specifications:', data.specifications);
    console.log('✓ Features:', data.features);

    // Create the popup
    const popup = document.createElement('div');
    popup.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 600px;
        max-height: 80vh;
        background: white;
        border: 3px solid #004990;
        border-radius: 12px;
        padding: 25px;
        z-index: 999999;
        overflow: auto;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
    `;

    const jsonString = JSON.stringify(data, null, 2);

    popup.innerHTML = `
        <div style="margin-bottom: 15px;">
            <h2 style="margin: 0 0 10px 0; color: #004990; font-size: 1.5em;">
                🧊 Refrigerator Data Extracted
            </h2>
            <div style="background: #f0f0f0; padding: 12px; border-radius: 6px; margin-bottom: 15px;">
                <strong>Brand:</strong> ${data.brand || 'N/A'}<br>
                <strong>Model:</strong> ${data.model || 'N/A'}<br>
                <strong>Capacity:</strong> ${data.capacity || 'N/A'}<br>
                <strong>Price:</strong> ${data.price || 'N/A'}<br>
                <strong>Dimensions:</strong>
                ${data.dimensions.width || '?'} W ×
                ${data.dimensions.height || '?'} H ×
                ${data.dimensions.depth || '?'} D
            </div>
        </div>
        <div style="margin-bottom: 15px;">
            <button id="copyBtn" style="padding: 12px 24px; margin-right: 10px; cursor: pointer; background: #004990; color: white; border: none; border-radius: 6px; font-size: 1em; font-weight: bold;">
                📋 Copy JSON
            </button>
            <button id="copyFormatted" style="padding: 12px 24px; margin-right: 10px; cursor: pointer; background: #28a745; color: white; border: none; border-radius: 6px; font-size: 1em; font-weight: bold;">
                📄 Copy Text Format
            </button>
            <button id="closeBtn" style="padding: 12px 24px; cursor: pointer; background: #666; color: white; border: none; border-radius: 6px; font-size: 1em;">
                ✕ Close
            </button>
        </div>
        <div style="margin-top: 15px;">
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 12px; font-size: 0.85em;">
                <strong>Instructions:</strong>
                <ol style="margin: 8px 0; padding-left: 20px;">
                    <li>Click "Copy JSON" to copy the structured data</li>
                    <li>Or click "Copy Text Format" for a readable format</li>
                    <li>Save to a file or send to someone</li>
                </ol>
            </div>
        </div>
        <pre style="margin-top: 15px; background: #f5f5f5; padding: 15px; border-radius: 6px; max-height: 300px; overflow: auto; font-size: 0.85em; border: 1px solid #ddd;">${jsonString}</pre>
    `;

    // Add backdrop
    const backdrop = document.createElement('div');
    backdrop.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 999998;
    `;

    document.body.appendChild(backdrop);
    document.body.appendChild(popup);

    // Copy JSON button
    document.getElementById('copyBtn').addEventListener('click', () => {
        navigator.clipboard.writeText(jsonString).then(() => {
            alert('✓ JSON data copied to clipboard!');
        });
    });

    // Copy formatted text button
    document.getElementById('copyFormatted').addEventListener('click', () => {
        const formatted = `
URL: ${data.url}

price: ${data.price}

Brand: ${data.brand}
Model: ${data.model}
Capacity: ${data.capacity}

Dimensions:
- Width: ${data.dimensions.width || 'N/A'}
- Height: ${data.dimensions.height || 'N/A'}
- Depth: ${data.dimensions.depth || 'N/A'}
- Depth (no handles): ${data.dimensions.depth_no_handles || 'N/A'}

Specifications:
${Object.entries(data.specifications).map(([k, v]) => `- ${k}: ${v}`).join('\n')}

Features:
${data.features.map(f => `- ${f}`).join('\n')}
        `.trim();

        navigator.clipboard.writeText(formatted).then(() => {
            alert('✓ Formatted text copied to clipboard!');
        });
    });

    // Close button
    document.getElementById('closeBtn').addEventListener('click', () => {
        popup.remove();
        backdrop.remove();
    });

    // Close on backdrop click
    backdrop.addEventListener('click', () => {
        popup.remove();
        backdrop.remove();
    });

    console.log('✅ Extraction complete! See popup window.');
})();
