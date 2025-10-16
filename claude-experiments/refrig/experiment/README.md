# Lowe's Product Scraper

A Playwright-based web scraper for extracting product information from Lowe's product pages.

## Installation

```bash
npm install
npx playwright install chromium
```

## Usage

Run the scraper with the default URL:

```bash
node scraper.js
```

Or provide a custom Lowe's product URL:

```bash
node scraper.js "https://www.lowes.com/pd/Product-Name/12345678"
```

## Features

The scraper extracts the following information:
- Product title
- Price
- Model number
- SKU/Item number
- Brand
- Rating and review count
- Description
- Specifications
- Product images
- Availability status

## Output

The scraped data is output as JSON to the console, including:
- All product details
- Original URL
- Timestamp of when the data was scraped

## Configuration

You can modify `headless` mode in the script:
- `headless: false` - See the browser while scraping (useful for debugging)
- `headless: true` - Run without visible browser (faster, for production)

## Notes

- The scraper includes a realistic user agent to avoid detection
- Waits for network idle to ensure all content is loaded
- Handles various selector patterns to accommodate Lowe's page structure
- Includes error handling and graceful shutdown
