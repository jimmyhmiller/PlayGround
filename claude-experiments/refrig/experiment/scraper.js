const { chromium } = require('playwright');

async function scrapeLowesProduct(url) {
  // Launch browser
  const browser = await chromium.launch({
    headless: false // Set to true for production
  });

  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
  });

  const page = await context.newPage();

  try {
    console.log(`Navigating to ${url}...`);
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });

    // Wait for key elements to load
    await page.waitForSelector('h1', { timeout: 10000 });

    // Extract product information
    const productData = await page.evaluate(() => {
      const data = {};

      // Product title
      const titleElement = document.querySelector('h1');
      data.title = titleElement ? titleElement.textContent.trim() : null;

      // Price
      const priceElement = document.querySelector('[data-selector="ia-price"]') ||
                          document.querySelector('.price-display') ||
                          document.querySelector('[class*="price"]');
      data.price = priceElement ? priceElement.textContent.trim() : null;

      // Model number
      const modelElement = document.querySelector('[data-selector="model-number"]') ||
                          Array.from(document.querySelectorAll('span')).find(el =>
                            el.textContent.includes('Model #'));
      data.model = modelElement ? modelElement.textContent.replace('Model #', '').trim() : null;

      // SKU/Item number
      const itemElement = document.querySelector('[data-selector="item-number"]') ||
                         Array.from(document.querySelectorAll('span')).find(el =>
                           el.textContent.includes('Item #'));
      data.item = itemElement ? itemElement.textContent.replace('Item #', '').trim() : null;

      // Brand
      const brandElement = document.querySelector('[itemprop="brand"]') ||
                          document.querySelector('[data-selector="brand"]');
      data.brand = brandElement ? brandElement.textContent.trim() : null;

      // Rating
      const ratingElement = document.querySelector('[class*="rating"]') ||
                           document.querySelector('[itemprop="ratingValue"]');
      data.rating = ratingElement ? ratingElement.textContent.trim() : null;

      // Number of reviews
      const reviewsElement = document.querySelector('[class*="review-count"]') ||
                            document.querySelector('[itemprop="reviewCount"]');
      data.reviewCount = reviewsElement ? reviewsElement.textContent.trim() : null;

      // Description
      const descElement = document.querySelector('[data-selector="product-description"]') ||
                         document.querySelector('.product-description') ||
                         document.querySelector('[itemprop="description"]');
      data.description = descElement ? descElement.textContent.trim() : null;

      // Specifications
      const specs = {};
      const specItems = document.querySelectorAll('[class*="spec"]') ||
                       document.querySelectorAll('.specifications li');
      specItems.forEach(item => {
        const label = item.querySelector('[class*="label"]') || item.querySelector('dt');
        const value = item.querySelector('[class*="value"]') || item.querySelector('dd');
        if (label && value) {
          specs[label.textContent.trim()] = value.textContent.trim();
        }
      });
      data.specifications = specs;

      // Images
      const images = [];
      const imageElements = document.querySelectorAll('[data-selector="product-image"]') ||
                           document.querySelectorAll('.product-image img') ||
                           document.querySelectorAll('[class*="gallery"] img');
      imageElements.forEach(img => {
        const src = img.src || img.dataset.src;
        if (src && !images.includes(src)) {
          images.push(src);
        }
      });
      data.images = images;

      // Availability
      const availabilityElement = document.querySelector('[class*="availability"]') ||
                                 document.querySelector('[data-selector="availability"]');
      data.availability = availabilityElement ? availabilityElement.textContent.trim() : null;

      return data;
    });

    // Add the URL to the data
    productData.url = url;
    productData.scrapedAt = new Date().toISOString();

    console.log('\n=== Scraped Product Data ===');
    console.log(JSON.stringify(productData, null, 2));

    return productData;

  } catch (error) {
    console.error('Error scraping product:', error.message);
    throw error;
  } finally {
    await browser.close();
  }
}

// Main execution
if (require.main === module) {
  const url = process.argv[2] ||
    'https://www.lowes.com/pd/Samsung-Counter-depth-Mega-Capacity-25-5-cu-ft-Smart-French-Door-Refrigerator-with-Dual-Ice-Maker-Water-and-Ice-Dispenser-Fingerprint-Resistant-Stainless-Steel-ENERGY-STAR/5015371771';

  scrapeLowesProduct(url)
    .then(() => {
      console.log('\nScraping completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { scrapeLowesProduct };
