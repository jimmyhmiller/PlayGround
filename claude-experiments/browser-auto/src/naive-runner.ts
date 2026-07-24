import type { Browser } from "playwright";

/**
 * The comparison arm for the timing-independence property: the same buy
 * journey written the way e2e tests are commonly written in raw Playwright —
 * good semantic locators, but timing encoded as fixed tolerances (explicit
 * timeouts around each assertion). This is NOT a strawman: every wait is an
 * explicit, generous-looking 1.5s, the pattern most suites converge on.
 *
 * The point: its outcome is a FUNCTION OF THE APP'S TIMING PROFILE. bat's
 * outcome, on the same journey and profiles, is not.
 */
export async function naiveBuyJourney(browser: Browser, baseUrl: string): Promise<{ ok: boolean; failedAt: string }> {
  const context = await browser.newContext();
  const page = await context.newPage();
  const TOLERANCE = 1500; // "should be plenty"
  try {
    const step = async (label: string, fn: () => Promise<void>) => {
      try {
        await fn();
        return null;
      } catch {
        return label;
      }
    };

    let failedAt =
      (await step("load products", async () => {
        await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
        await page.getByRole("heading", { name: "Products" }).waitFor({ timeout: TOLERANCE });
      })) ??
      (await step("add blue widget", async () => {
        await page
          .getByRole("listitem")
          .filter({ hasText: "Blue Widget" })
          .getByRole("button", { name: "Add to cart" })
          .click({ timeout: TOLERANCE });
        await page.getByRole("status").waitFor({ timeout: TOLERANCE }); // the toast
        await page.locator('[data-testid="cart-count"]', { hasText: "1" }).waitFor({ timeout: TOLERANCE });
      })) ??
      (await step("add red widget", async () => {
        await page
          .getByRole("listitem")
          .filter({ hasText: "Red Widget" })
          .getByRole("button", { name: "Add to cart" })
          .click({ timeout: TOLERANCE });
        await page.locator('[data-testid="cart-count"]', { hasText: "2" }).waitFor({ timeout: TOLERANCE });
      })) ??
      (await step("open cart", async () => {
        await page.getByRole("link", { name: "Cart" }).click({ timeout: TOLERANCE });
        await page.getByRole("row", { name: /Blue Widget/ }).waitFor({ timeout: TOLERANCE });
      }));

    return { ok: failedAt === null, failedAt: failedAt ?? "" };
  } finally {
    await context.close();
  }
}
