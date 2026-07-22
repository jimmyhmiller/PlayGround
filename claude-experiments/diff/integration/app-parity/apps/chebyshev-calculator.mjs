// chebyshev-calculator — antd UI that recomputes Chebyshev approximation
// coefficients and redraws two <canvas> plots on every parameter change.
// Coefficient float formatting in the DOM is a strong numerical-parity signal;
// the canvases are compared through the screenshot channel.
import { sleep } from "../harness.mjs";

const APP =
  "/tmp/claude-1000/-home-jimmyhmiller-Documents-Code-Playground/19134627-eb88-4448-8dc7-b23ce19dea39/scratchpad/oss-triage/chebyshev-calculator";

export default {
  name: "chebyshev-calculator",
  appDir: APP,
  base: "/",
  notes: [
    "Canvas plots cannot be compared via DOM/styles; they are covered by the full-page screenshot channel (canvases are same-origin, untainted).",
  ],
  steps: [
    {
      name: "initial-load",
      settle: 900,
      run: async (page, ctx) => {
        await page.goto(ctx.url, { waitUntil: "networkidle0", timeout: 30000 });
        await page.waitForSelector("canvas", { timeout: 15000 });
        await page.waitForFunction(
          () => document.body.innerText.includes("Coefficients"),
          { timeout: 15000 }
        );
      },
    },
    {
      name: "slider-increase-terms",
      settle: 900,
      run: async (page) => {
        // antd Slider: focus the handle and bump the value by keyboard —
        // deterministic (no pointer-position dependence).
        await page.focus(".ant-slider-handle");
        await page.keyboard.press("ArrowRight");
      },
      probe: async (page) =>
        page.evaluate(() => ({
          slider:
            document
              .querySelector(".ant-slider-handle")
              ?.getAttribute("aria-valuenow") ?? null,
          coefficientCount: document.body.innerText.match(/c\d+/g)?.length ?? 0,
        })),
    },
    {
      name: "edit-xmax",
      settle: 900,
      run: async (page) => {
        // Third text input is xmax (f(x), xmin, xmax).
        const inputs = await page.$$("input.ant-input");
        const xmax = inputs[2];
        await xmax.click({ clickCount: 3 }); // select current value
        await page.keyboard.type("5", { delay: 20 });
        await sleep(200);
      },
      probe: async (page) =>
        page.evaluate(
          () => document.querySelectorAll("input.ant-input")[2]?.value ?? null
        ),
    },
    {
      name: "toggle-first-checkbox",
      settle: 900,
      run: async (page) => {
        const box = await page.$(".ant-checkbox-input");
        await box.click();
      },
      probe: async (page) =>
        page.evaluate(() =>
          [...document.querySelectorAll(".ant-checkbox-input")].map((c) => c.checked)
        ),
    },
    {
      name: "switch-match-segment",
      settle: 900,
      run: async (page) => {
        // Segmented "Match" control: click the second segment label.
        await page.evaluate(() => {
          const items = document.querySelectorAll(".ant-segmented-item");
          if (items.length > 1) items[1].querySelector("input")?.click();
        });
      },
      probe: async (page) =>
        page.evaluate(() =>
          [...document.querySelectorAll(".ant-segmented-item-input")].map(
            (r) => r.checked
          )
        ),
    },
  ],
};
