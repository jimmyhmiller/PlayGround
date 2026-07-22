// the-last-pawn — chess-survival game (redux + framer-motion + CSS-module
// SCSS), built for base /the-last-pawn/. Despite the name "board cells", the
// game is driven by keyboard (w/a/s/d to move, per src/features/game/hooks/
// useInputs.js) — clicking a GridCell does nothing, so moves are made with the
// keyboard. Enemy spawning/movement uses Math.random, which the harness seeds
// identically on both sides, making full gameplay exactly comparable.
import { clickText, sleep } from "../harness.mjs";

const APP =
  "/tmp/claude-1000/-home-jimmyhmiller-Documents-Code-Playground/19134627-eb88-4448-8dc7-b23ce19dea39/scratchpad/oss-triage/the-last-pawn";

export default {
  name: "the-last-pawn",
  appDir: APP,
  base: "/the-last-pawn/",
  notes: [
    "Moves are keyboard-driven (w/a/s/d), not cell clicks — see useInputs.js.",
    "All game randomness goes through the page's Math.random, seeded identically on both sides by the harness, so spawns/AI are exactly comparable.",
    "framer-motion animates via JS (inline styles); each step waits for the animation to settle before capture.",
  ],
  steps: [
    {
      name: "initial-load-menu",
      settle: 1200,
      run: async (page, ctx) => {
        await page.goto(ctx.url, { waitUntil: "networkidle0", timeout: 30000 });
        await page.waitForFunction(
          () => document.body.innerText.includes("PLAY"),
          { timeout: 15000 }
        );
      },
    },
    {
      name: "how-to-play",
      settle: 1200,
      run: async (page) => {
        await clickText(page, "HOW TO PLAY");
        await page.waitForFunction(
          () => document.body.innerText.includes("BACK"),
          { timeout: 15000 }
        );
      },
    },
    {
      name: "back-to-menu",
      // The gear page-transition overlay (framer-motion) runs ~2s; wait for it
      // to fully unmount so both sides capture the settled menu.
      settle: 3000,
      run: async (page) => {
        await clickText(page, "BACK");
        await page.waitForFunction(
          () => document.body.innerText.includes("PLAY"),
          { timeout: 15000 }
        );
      },
    },
    {
      name: "start-game",
      settle: 1500,
      run: async (page) => {
        await clickText(page, "PLAY");
        await page.waitForFunction(
          () => document.body.innerText.includes("SCORE"),
          { timeout: 15000 }
        );
      },
    },
    {
      name: "move-right",
      settle: 1400,
      run: async (page) => {
        await page.keyboard.down("d");
        await sleep(80);
        await page.keyboard.up("d");
      },
      probe: async (page) =>
        page.evaluate(() => document.body.innerText.replace(/\s+/g, " ").trim()),
    },
    {
      name: "move-up",
      settle: 1400,
      run: async (page) => {
        await page.keyboard.down("w");
        await sleep(80);
        await page.keyboard.up("w");
      },
      probe: async (page) =>
        page.evaluate(() => document.body.innerText.replace(/\s+/g, " ").trim()),
    },
    {
      name: "move-right-again",
      settle: 1400,
      run: async (page) => {
        await page.keyboard.down("d");
        await sleep(80);
        await page.keyboard.up("d");
      },
      probe: async (page) =>
        page.evaluate(() => document.body.innerText.replace(/\s+/g, " ").trim()),
    },
  ],
};
