/** Open the bat ui, open the failing run, open a passing-rerun diff, screenshot. */
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
await page.goto("http://localhost:8123/");
await page.getByRole("button", { name: /failed @ step/ }).first().click();
await page.locator("h1", { hasText: "flaky cart badge" }).waitFor({ timeout: 5000 });
const passRerun = page.getByRole("button", { name: /rerun \d+ \(pass\)/ }).first();
await passRerun.click();
await page.locator(".diff-grid").waitFor({ timeout: 5000 });
await page.locator(".diff-wrap").scrollIntoViewIfNeeded();
await page.screenshot({ path: "/tmp/bat-ui-diff.png", clip: (await page.locator(".diff-wrap").boundingBox())! });
console.log("diff screenshot saved");
await browser.close();
process.exit(0);
