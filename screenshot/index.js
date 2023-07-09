import puppeteer from "puppeteer";


const browser = await puppeteer.launch({
	headless: "new"
});

const page = await browser.newPage();


const websiteUrl = process.argv[2];
console.log("website", websiteUrl)

// Open URL in current page  
await page.goto(websiteUrl, { waitUntil: 'networkidle0' });

await page.screenshot({
  path: 'screenshot.jpg'
});

process.exit()