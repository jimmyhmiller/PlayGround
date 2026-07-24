import { createServer, type Server } from "node:http";
import { readdir, readFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { BatConfig } from "../config.js";
import { loadSeeds, loadWorldHandle, resolveWorkerConfig } from "../config.js";
import { launchApp } from "../runner/appenv.js";
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
          const reruns: Array<{ name: string; status: string }> = [];
          for (const name of (await readdir(join(dir, "reruns")).catch(() => [] as string[])).sort()) {
            if (!SAFE.test(name) || !name.endsWith(".json")) continue;
            try {
              const rt = JSON.parse(await readFile(join(dir, "reruns", name), "utf8")) as FlowTrace;
              reruns.push({ name, status: rt.status });
            } catch {
              // partial file
            }
          }
          return json(200, { trace, report, reruns });
        }
        if (url.pathname === "/api/rerun") {
          const slug = url.searchParams.get("flow") ?? "";
          const id = url.searchParams.get("run") ?? "";
          const name = basename(url.searchParams.get("name") ?? "");
          if (!SAFE.test(slug) || !SAFE.test(id) || !SAFE.test(name) || !name.endsWith(".json")) {
            return json(400, { error: "bad params" });
          }
          const rt = JSON.parse(await readFile(join(config.root, ".bat", "runs", slug, id, "reruns", name), "utf8")) as FlowTrace;
          return json(200, rt);
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
          // configs with an `app` spec expect bat to launch the app — lease one instance for the replay
          let replayConfig = config;
          let app: Awaited<ReturnType<typeof launchApp>> | null = null;
          if (config.app) {
            app = await launchApp(config.app, 0, config.baseUrl, config.root);
            replayConfig = resolveWorkerConfig(config, { port: app.port, index: 0 });
          }
          const seeds = await loadSeeds(replayConfig);
          const world = await loadWorldHandle(replayConfig, { index: 0, baseUrl: replayConfig.baseUrl, port: app?.port ?? null });
          const browser = await launchBrowser({ ...replayConfig, headless: !body.headed });
          try {
            const result = await replayStep(body.file, body.step, { config: replayConfig, world, seeds, browser }, { fast: false });
            return json(200, { tier: result.tier, status: result.trace.status, report: renderReport(result.trace) });
          } finally {
            await browser.close();
            await app?.stop();
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
