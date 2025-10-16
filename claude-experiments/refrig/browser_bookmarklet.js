/**
 * REFRIGERATOR DATA EXTRACTOR BOOKMARKLET
 *
 * To use this:
 * 1. Create a new bookmark in your browser
 * 2. Name it "Extract Fridge Data"
 * 3. In the URL field, paste this entire code wrapped in: javascript:(function(){...})();
 * 4. Visit each refrigerator page
 * 5. Click the bookmarklet
 * 6. Copy the JSON output from the console or alert
 *
 * OR run this directly in the browser console (F12) on each page
 */

(function() {
    function extractRefrigeratorData() {
        const url = window.location.href;
        const isHomeDepot = url.includes('homedepot.com');
        const isLowes = url.includes('lowes.com');

        const data = {
            url: url,
            store: isHomeDepot ? 'Home Depot' : (isLowes ? "Lowe's" : 'Unknown'),
            title: document.title,
            model: null,
            brand: null,
            capacity: null,
            dimensions: {},
            price: null,
            features: [],
            images: [],
            specs: {}
        };

        // Extract price
        const priceSelectors = [
            '[data-testid="product-price"]',
            '.price-format__main-price',
            '.price',
            '[itemprop="price"]',
            '.product-price',
            '[data-component="PriceDisplay"]'
        ];

        for (const selector of priceSelectors) {
            const priceElem = document.querySelector(selector);
            if (priceElem) {
                const priceText = priceElem.textContent;
                const priceMatch = priceText.match(/\$[\d,]+\.?\d*/);
                if (priceMatch) {
                    data.price = priceMatch[0];
                    break;
                }
            }
        }

        // Extract from page text
        const bodyText = document.body.innerText;

        // Model number
        const modelMatch = bodyText.match(/Model[#\s:]+([A-Z0-9-]+)/i);
        if (modelMatch) {
            data.model = modelMatch[1];
        }

        // Capacity from title or body
        const capacityMatch = bodyText.match(/(\d+\.?\d*)\s*cu\.?\s*ft/i);
        if (capacityMatch) {
            data.capacity = `${capacityMatch[1]} cu ft`;
        }

        // Brand from title
        const brandMatch = data.title.match(/^(\w+)/);
        if (brandMatch) {
            data.brand = brandMatch[1];
        }

        // Try to find specifications table or list
        const specTables = document.querySelectorAll('table, dl, [class*="spec"]');

        specTables.forEach(table => {
            const text = table.innerText.toLowerCase();

            // Height
            const heightMatch = text.match(/height.*?(\d+\.?\d*)\s*(?:in|inches)/i);
            if (heightMatch && !data.dimensions.height) {
                data.dimensions.height = `${heightMatch[1]} inches`;
            }

            // Width
            const widthMatch = text.match(/width.*?(\d+\.?\d*)\s*(?:in|inches)/i);
            if (widthMatch && !data.dimensions.width) {
                data.dimensions.width = `${widthMatch[1]} inches`;
            }

            // Depth
            const depthMatch = text.match(/depth.*?(\d+\.?\d*)\s*(?:in|inches)/i);
            if (depthMatch && !data.dimensions.depth) {
                data.dimensions.depth = `${depthMatch[1]} inches`;
            }
        });

        // Get product images
        const imgElements = document.querySelectorAll('img[src*="product"], img[src*="media"], img[class*="product"]');
        imgElements.forEach((img, index) => {
            if (index < 5 && img.src && img.src.startsWith('http')) {
                data.images.push(img.src);
            }
        });

        // Look for key features
        const featureElements = document.querySelectorAll('[class*="feature"], [class*="bullet"], li');
        featureElements.forEach((elem, index) => {
            if (index < 10) {
                const text = elem.textContent.trim();
                if (text.length > 10 && text.length < 200 && !text.includes('©')) {
                    data.features.push(text);
                }
            }
        });

        return data;
    }

    // Extract the data
    const fridgeData = extractRefrigeratorData();

    // Display results
    console.log('=== REFRIGERATOR DATA ===');
    console.log(JSON.stringify(fridgeData, null, 2));
    console.log('=== COPY THE JSON ABOVE ===');

    // Create a popup with the data
    const jsonString = JSON.stringify(fridgeData, null, 2);
    const popup = document.createElement('div');
    popup.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        width: 500px;
        max-height: 80vh;
        background: white;
        border: 3px solid #333;
        border-radius: 8px;
        padding: 20px;
        z-index: 999999;
        overflow: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-family: monospace;
        font-size: 12px;
    `;

    popup.innerHTML = `
        <div style="margin-bottom: 10px; font-weight: bold; font-size: 14px;">
            Refrigerator Data Extracted
        </div>
        <div style="margin-bottom: 10px;">
            <strong>Store:</strong> ${fridgeData.store}<br>
            <strong>Brand:</strong> ${fridgeData.brand || 'N/A'}<br>
            <strong>Model:</strong> ${fridgeData.model || 'N/A'}<br>
            <strong>Capacity:</strong> ${fridgeData.capacity || 'N/A'}<br>
            <strong>Price:</strong> ${fridgeData.price || 'N/A'}<br>
            <strong>Dimensions:</strong>
            ${fridgeData.dimensions.width || '?'} W x
            ${fridgeData.dimensions.height || '?'} H x
            ${fridgeData.dimensions.depth || '?'} D
        </div>
        <button id="copyBtn" style="padding: 8px 16px; margin-right: 8px; cursor: pointer; background: #0078d4; color: white; border: none; border-radius: 4px;">
            Copy JSON
        </button>
        <button id="closeBtn" style="padding: 8px 16px; cursor: pointer; background: #ccc; border: none; border-radius: 4px;">
            Close
        </button>
        <pre style="margin-top: 10px; background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 300px; overflow: auto;">${jsonString}</pre>
    `;

    document.body.appendChild(popup);

    // Copy button functionality
    document.getElementById('copyBtn').addEventListener('click', () => {
        navigator.clipboard.writeText(jsonString).then(() => {
            alert('JSON copied to clipboard!');
        });
    });

    // Close button functionality
    document.getElementById('closeBtn').addEventListener('click', () => {
        popup.remove();
    });
})();
