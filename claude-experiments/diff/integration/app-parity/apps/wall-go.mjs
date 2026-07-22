// wall-go — Devil's-Plan-style board game (react + zustand + AI in a module
// Web Worker), built for base /wall-go/. Starting a vs-AI game and getting an
// AI reply proves the emitted worker bundle boots AND plays.
//
// The AI's move choice goes through Math.random *inside the Worker*
// (src/utils/ai.ts random tie-break, src/agents/MinimaxAgent.ts random think
// delay). The harness's Math.random seeding only reaches the main thread, so
// the AI's placement square is genuinely nondeterministic across the two
// sides -> the AI-reply step is `mode: "invariant"` (see `why` below).
const APP =
  "/tmp/claude-1000/-home-jimmyhmiller-Documents-Code-Playground/19134627-eb88-4448-8dc7-b23ce19dea39/scratchpad/oss-triage/wall-go";

const boardStateFn = () => {
  const cells = [...document.querySelectorAll("[data-cell-x]")];
  const stones = { R: [], B: [] };
  for (const c of cells) {
    const x = c.getAttribute("data-cell-x");
    const y = c.getAttribute("data-cell-y");
    if (c.querySelector("button.bg-rose-500")) stones.R.push(`${x},${y}`);
    else if (c.querySelector("button.bg-indigo-500")) stones.B.push(`${x},${y}`);
  }
  stones.R.sort();
  stones.B.sort();
  return { cellCount: cells.length, R: stones.R, B: stones.B };
};

export default {
  name: "wall-go",
  appDir: APP,
  base: "/wall-go/",
  initStorage: {
    // Pin theme + language so first-visit detection (prefers-color-scheme /
    // navigator.language) can't drift between runs.
    theme: "light",
    i18nextLng: "en",
  },
  notes: [
    "AI runs in a module Web Worker; the ai-reply step proves worker behavioral parity (boots, receives game state, replies with a legal placement).",
    "Human plays Red and moves first ('First' button sets the AI to Blue).",
  ],
  steps: [
    {
      name: "initial-load-menu",
      settle: 800,
      run: async (page, ctx) => {
        await page.goto(ctx.url, { waitUntil: "networkidle0", timeout: 30000 });
        await page.waitForFunction(
          () => document.body.innerText.includes("Wall Go"),
          { timeout: 15000 }
        );
      },
    },
    {
      name: "open-vs-ai-select",
      settle: 800,
      run: async (page) => {
        await page.evaluate(() => {
          const btn = [...document.querySelectorAll("button")].find((b) =>
            b.textContent.includes("vs AI")
          );
          btn.click();
        });
      },
    },
    {
      name: "choose-easy-level",
      settle: 500,
      run: async (page) => {
        await page.select("select", "easy");
      },
      probe: async (page) => page.evaluate(() => document.querySelector("select").value),
    },
    {
      name: "start-as-first",
      settle: 900,
      run: async (page) => {
        await page.evaluate(() => {
          const btn = [...document.querySelectorAll("button")].find((b) =>
            b.textContent.includes("First")
          );
          btn.click();
        });
        // Placement-phase board (7x7 = 49 cells) with human (Red) to act.
        await page.waitForFunction(
          () => document.querySelectorAll("[data-cell-x]").length === 49,
          { timeout: 15000 }
        );
      },
      probe: async (page) => page.evaluate(boardStateFn),
    },
    {
      name: "place-stone-and-await-ai-reply",
      mode: "invariant",
      why:
        "The AI's placement square is chosen with Math.random INSIDE the AI Web Worker (utils/ai.ts tie-break + MinimaxAgent think-delay); init scripts cannot seed worker RNG, so the exact square (and thus DOM/pixels) legitimately differs between the two sides. We assert identical INVARIANTS instead: our Red stone lands on the clicked square, exactly one Blue (AI) stone appears on a legal empty square, stone counts match, and neither side logs errors.",
      settle: 600,
      run: async (page) => {
        // Board opens with 4 pre-placed stones: R at (1,1),(5,5); B at
        // (1,5),(5,1). Place the human stone on the center cell (3,3); the
        // placing order then gives the AI (Blue) two consecutive placements.
        await page.evaluate(() => {
          const cell = document.querySelector("[data-cell-x='3'][data-cell-y='3']");
          const btn = cell.querySelector("button");
          btn.click();
        });
        // Wait for the AI worker's replies: Blue goes from 2 -> 4 stones.
        await page.waitForFunction(
          () =>
            [...document.querySelectorAll("[data-cell-x]")].filter((c) =>
              c.querySelector("button.bg-indigo-500")
            ).length >= 4,
          { timeout: 30000 }
        );
      },
      check: async ({ refPage, diffPage, refSnap, diffSnap }) => {
        const ref = await refPage.evaluate(boardStateFn);
        const diff = await diffPage.evaluate(boardStateFn);
        const errsOf = (snap) =>
          snap.net.console.filter(
            (m) => m.type === "error" || m.type === "pageerror"
          );
        const refErrs = errsOf(refSnap);
        const diffErrs = errsOf(diffSnap);
        const legalSquare = (s) => {
          const [x, y] = s.split(",").map(Number);
          return x >= 0 && x < 7 && y >= 0 && y < 7;
        };
        const RED_EXPECTED = "1,1|3,3|5,5"; // 2 pre-placed + our (3,3) click
        const PREPLACED_B = ["1,5", "5,1"];
        const aiPlacementsOk = (side) =>
          side.B.length === 4 &&
          PREPLACED_B.every((s) => side.B.includes(s)) &&
          side.B.every(
            (s) => legalSquare(s) && !side.R.includes(s)
          );
        return [
          {
            name: "red stones deterministic: pre-placed + our (3,3) click (both sides)",
            pass:
              ref.R.join("|") === RED_EXPECTED &&
              diff.R.join("|") === RED_EXPECTED,
            detail: `ref.R=${ref.R} diffpack.R=${diff.R}`,
          },
          {
            name: "AI placed exactly two stones, on legal empty squares (both sides)",
            pass: aiPlacementsOk(ref) && aiPlacementsOk(diff),
            detail: `ref.B=${ref.B} diffpack.B=${diff.B}`,
          },
          {
            name: "board structure identical (49 cells both sides)",
            pass: ref.cellCount === 49 && diff.cellCount === 49,
            detail: `ref=${ref.cellCount} diffpack=${diff.cellCount}`,
          },
          {
            name: "zero console/page errors on both sides",
            pass: refErrs.length === 0 && diffErrs.length === 0,
            detail:
              `ref=[${refErrs.map((e) => e.text).join("; ")}] ` +
              `diffpack=[${diffErrs.map((e) => e.text).join("; ")}]`,
          },
        ];
      },
    },
  ],
};
