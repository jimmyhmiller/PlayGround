// redux-essentials-example-app — Redux Toolkit tutorial app (this clone is at
// the tutorial's *starting point*: a single welcome route and an empty navbar
// with NO nav links, so there is nothing to click through; instead we exercise
// the MSW mock-API service worker boot + a controlled reload).
//
// Determinism: the app seeds @faker-js/faker from localStorage
// "randomTimestampSeed" (an ISO date). We preset the SAME fixed value on both
// sides so any faker-derived data/timestamps are identical.
import { sleep } from "../harness.mjs";

const APP =
  "/tmp/claude-1000/-home-jimmyhmiller-Documents-Code-Playground/19134627-eb88-4448-8dc7-b23ce19dea39/scratchpad/oss/redux-essentials-example-app";

export default {
  name: "redux-essentials",
  appDir: APP,
  base: "/",
  initStorage: {
    // Fixed faker seed date (see src/api/server.ts) — identical on both sides.
    randomTimestampSeed: "2026-01-01T00:00:00.000Z",
  },
  notes: [
    "This clone is the tutorial starting point: Navbar has no links (src/components/Navbar.tsx renders an empty .navLinks). 'Click through nav links' is therefore vacuous; documented instead of faked.",
    "MSW's mockServiceWorker.js registers on first load; the reload step exercises the service-worker-controlled fetch path on both sides.",
  ],
  steps: [
    {
      name: "initial-load",
      settle: 800,
      run: async (page, ctx) => {
        await page.goto(ctx.url, { waitUntil: "networkidle0", timeout: 30000 });
        // Welcome route content
        await page.waitForFunction(
          () => document.body.innerText.includes("Welcome to the Redux Essentials"),
          { timeout: 15000 }
        );
      },
      probe: async (page) =>
        page.evaluate(async () => ({
          swRegistered: !!(await navigator.serviceWorker.getRegistration()),
          h1: document.querySelector("h1")?.textContent ?? null,
          h2: document.querySelector("h2")?.textContent ?? null,
        })),
    },
    {
      name: "reload-under-service-worker",
      settle: 800,
      run: async (page) => {
        // Wait until the SW is activated so the reload is SW-controlled on
        // both sides (removes a boot race from the comparison).
        await page.evaluate(() => navigator.serviceWorker.ready.then(() => {}));
        await page.reload({ waitUntil: "networkidle0", timeout: 30000 });
        await page.waitForFunction(
          () => document.body.innerText.includes("Welcome to the Redux Essentials"),
          { timeout: 15000 }
        );
      },
      probe: async (page) =>
        page.evaluate(() => ({
          controlled: !!navigator.serviceWorker.controller,
          title: document.title,
        })),
    },
    {
      name: "client-side-navigation-404-route",
      settle: 500,
      run: async (page) => {
        // No nav links exist; drive react-router directly to a non-matching
        // route and back, proving the router bundle behaves identically.
        await page.evaluate(() => {
          window.history.pushState({}, "", "/no-such-route");
          window.dispatchEvent(new PopStateEvent("popstate"));
        });
        await sleep(300);
        await page.evaluate(() => {
          window.history.pushState({}, "", "/");
          window.dispatchEvent(new PopStateEvent("popstate"));
        });
      },
      probe: async (page) =>
        page.evaluate(() => ({
          path: location.pathname,
          welcomeVisible: document.body.innerText.includes(
            "Welcome to the Redux Essentials"
          ),
        })),
    },
  ],
};
