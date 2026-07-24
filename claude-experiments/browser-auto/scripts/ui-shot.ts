/** Open the bat ui, click into the latest failing run, screenshot it. */
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1400 } });
await page.goto("http://localhost:8123/");
await page.getByRole("button", { name: /failed @ step/ }).first().click();
await page.locator("h1", { hasText: "flaky cart badge" }).waitFor({ timeout: 5000 });
await page.waitForTimeout(800); // deliberate: probe — let the screenshot img load
await page.screenshot({ path: "/tmp/bat-ui-screenshot.png", fullPage: true });
console.log("screenshot saved");
await browser.close();
process.exit(0);
