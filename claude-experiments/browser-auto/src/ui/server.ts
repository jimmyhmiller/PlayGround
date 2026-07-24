import { createServer, type Server } from "node:http";
import { readdir, readFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { BatConfig } from "../config.js";
import { loadSeeds, loadWorldHandle } from "../config.js";
import { launchBrowser, replayStep } from "../runner/run.js";
import { renderReport, type FlowTrace } from "../runner/trace.js";

const STATIC_DIR = join(dirname(fileURLToPath(import.meta.url)), "static");

const SAFE = /^[\w.-]+$/;

interface RunSummary {
  id: string;
  flowName: string;
  status: string;
  startedAt: string;
  failedStep: number | null;
  file: string;
}

async function listRuns(root: string): Promise<Array<{ slug: string; runs: RunSummary[] }>> {
  const base = join(root, ".bat", "runs");
  const out: Array<{ slug: string; runs: RunSummary[] }> = [];
  for (const slug of (await readdir(base).catch(() => [] as string[])).sort()) {
    if (!SAFE.test(slug) || slug === "latest") continue;
    const runs: RunSummary[] = [];
    for (const id of (await readdir(join(base, slug)).catch(() => [] as string[])).sort().reverse()) {
      if (!SAFE.test(id) || id === "latest") continue;
      try {
        const trace = JSON.parse(await readFile(join(base, slug, id, "trace.json"), "utf8")) as FlowTrace;
        const failed = trace.steps.find((s) => s.status === "fail");
        runs.push({
          id,
          flowName: trace.flow,
          status: trace.status,
          startedAt: trace.startedAt,
          failedStep: failed ? failed.index + 1 : null,
          file: trace.file,
        });
      } catch {
        // partial/interrupted run — skip
      }
    }
    if (runs.length) out.push({ slug, runs });
  }
  return out;
}

export async function startUiServer(config: BatConfig, port: number): Promise<Server> {
  const server = createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? "/", "http://localhost");
      const send = (status: number, type: string, body: string | Buffer) => {
        res.writeHead(status, { "content-type": type });
        res.end(body);
      };
      const json = (status: number, payload: unknown) => send(status, "application/json", JSON.stringify(payload));

      try {
        if (url.pathname === "/api/runs") {
          return json(200, await listRuns(config.root));
        }
        if (url.pathname === "/api/trace") {
          const slug = url.searchParams.get("flow") ?? "";
          const id = url.searchParams.get("run") ?? "";
          if (!SAFE.test(slug) || !SAFE.test(id)) return json(400, { error: "bad params" });
          const dir = join(config.root, ".bat", "runs", slug, id);
          const trace = JSON.parse(await readFile(join(dir, "trace.json"), "utf8")) as FlowTrace;
          const report = await readFile(join(dir, "report.txt"), "utf8").catch(() => renderReport(trace));
          return json(200, { trace, report });
        }
        if (url.pathname === "/api/artifact") {
          const slug = url.searchParams.get("flow") ?? "";
          const id = url.searchParams.get("run") ?? "";
          const name = basename(url.searchParams.get("name") ?? "");
          if (!SAFE.test(slug) || !SAFE.test(id) || !name.endsWith(".png")) return json(400, { error: "bad params" });
          const png = await readFile(join(config.root, ".bat", "runs", slug, id, name));
          return send(200, "image/png", png);
        }
        if (url.pathname === "/api/replay" && req.method === "POST") {
          const chunks: Buffer[] = [];
          for await (const c of req) chunks.push(c as Buffer);
          const body = JSON.parse(Buffer.concat(chunks).toString() || "{}") as { file?: string; step?: number; headed?: boolean };
          if (!body.file || !body.step) return json(400, { error: "file and step required" });
          const seeds = await loadSeeds(config);
          const world = await loadWorldHandle(config);
          const browser = await launchBrowser({ ...config, headless: !body.headed });
          try {
            const result = await replayStep(body.file, body.step, { config, world, seeds, browser }, { fast: false });
            return json(200, { tier: result.tier, status: result.trace.status, report: renderReport(result.trace) });
          } finally {
            await browser.close();
          }
        }

        // static
        const file = url.pathname === "/" ? "index.html" : basename(url.pathname);
        if (!SAFE.test(file)) return send(404, "text/plain", "not found");
        const type = file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html";
        const content = await readFile(join(STATIC_DIR, file));
        return send(200, type, content);
      } catch (e) {
        return json(500, { error: e instanceof Error ? e.message : String(e) });
      }
    })();
  });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return server;
}
