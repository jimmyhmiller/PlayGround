import { applyWorld, capabilityLevel, doctor } from "../world/adapter.js";
import { WorldError } from "../world/algebra.js";
import type { WorldAdapter, WorldDescription } from "../world/types.js";

/**
 * Mount the world adapter over HTTP so the runner can drive it from outside
 * the app's process. Returns a web-standard (Request) => Response handler —
 * usable directly as a Next.js App Router route handler:
 *
 *   // app/api/__bat/route.ts
 *   import { createWorldHandler } from "browser-auto/server";
 *   import world from "@/e2e/world";
 *   export const POST = createWorldHandler(world);
 *
 * The handler refuses to exist unless BAT_TEST=1 — it 404s so production
 * builds reveal nothing.
 */
export function createWorldHandler(adapter: WorldAdapter): (request: Request) => Promise<Response> {
  let lastIds: Record<string, Record<string, unknown>> = {};

  return async function batWorldHandler(request: Request): Promise<Response> {
    if (process.env.BAT_TEST !== "1") {
      return new Response("Not Found", { status: 404 });
    }
    let body: { op?: string; description?: WorldDescription; userKey?: string; id?: string };
    try {
      body = (await request.json()) as typeof body;
    } catch {
      return json({ ok: false, error: "body must be JSON" }, 400);
    }
    try {
      switch (body.op) {
        case "capabilities": {
          const { level } = capabilityLevel(adapter);
          return json({
            ok: true,
            result: {
              level,
              hasSession: !!adapter.session,
              hasSnapshot: !!adapter.snapshot,
              hasFingerprint: !!adapter.fingerprint,
            },
          });
        }
        case "apply": {
          if (!body.description) return json({ ok: false, error: "apply needs a description" }, 400);
          const applied = await applyWorld(adapter, body.description);
          lastIds = applied.ids as Record<string, Record<string, unknown>>;
          return json({ ok: true, result: { verification: applied.verification } });
        }
        case "session": {
          if (!adapter.session) {
            return json({ ok: false, problems: ["world adapter has no session() — add it to defineWorld"] }, 400);
          }
          const ctx = {
            id: (refOrType: unknown, key?: string) => {
              const [t, k] =
                typeof refOrType === "string"
                  ? [refOrType, key!]
                  : ((refOrType as { $ref: [string, string] }).$ref ?? [undefined, undefined]);
              const v = t !== undefined ? lastIds[t]?.[k!] : undefined;
              if (v === undefined) throw new WorldError([`session(): cannot resolve id for ${String(t)}/"${String(k)}"`]);
              return v;
            },
          };
          const session = await adapter.session(String(body.userKey), ctx as never);
          return json({ ok: true, result: session });
        }
        case "doctor":
          return json({ ok: true, result: doctor(adapter) });
        case "snapshot": {
          const id = adapter.snapshot ? await adapter.snapshot() : null;
          return json({ ok: true, result: id });
        }
        case "restore": {
          if (!adapter.restore) return json({ ok: false, problems: ["adapter has no restore()"] }, 400);
          await adapter.restore(String(body.id));
          return json({ ok: true, result: null });
        }
        case "fingerprint": {
          const fp = adapter.fingerprint ? await adapter.fingerprint() : null;
          return json({ ok: true, result: fp });
        }
        default:
          return json({ ok: false, error: `unknown op '${String(body.op)}'` }, 400);
      }
    } catch (e) {
      if (e instanceof WorldError) return json({ ok: false, problems: e.problems }, 422);
      return json({ ok: false, error: e instanceof Error ? e.message : String(e) }, 500);
    }
  };
}

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}
