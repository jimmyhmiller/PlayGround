import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { formatAction, formatEffect, ROLE_KINDS, type Action, type Effect, type Target } from "./ir.js";
import { parseFlow } from "./parser.js";

/**
 * Round-trip property: for ALL representable steps, parsing the canonical
 * text form reproduces the exact IR. The DSL's meaning can never depend on
 * how a flow happens to be written.
 */

const name = fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9 .,-]{0,18}[a-zA-Z0-9]$/);
const kind = fc.constantFrom(...ROLE_KINDS);
const path = fc
  .array(fc.stringMatching(/^[a-z][a-z0-9-]{0,6}$/), { minLength: 1, maxLength: 3 })
  .map((segs) => "/" + segs.join("/"));

const target: fc.Arbitrary<Target> = fc.letrec<{ target: Target }>((tie) => ({
  target: fc
    .tuple(
      fc.oneof(
        fc.record({ kind, name }, { requiredKeys: ["kind", "name"] }),
        fc.record({ kind: fc.constantFrom("testid", "field", "placeholder", "text") as fc.Arbitrary<Target["kind"]>, name }),
      ),
      fc.option(tie("target"), { nil: undefined, depthSize: "small" }),
    )
    .map(([base, within]) => (within ? ({ ...base, within } as Target) : (base as Target))),
})).target;

const action: fc.Arbitrary<Action> = fc.oneof(
  path.map((p): Action => ({ type: "go", path: p })),
  fc.tuple(fc.constantFrom("click", "dblclick", "hover") as fc.Arbitrary<"click">, target).map(
    ([type, t]): Action => ({ type, target: t }),
  ),
  fc.tuple(target, name).map(([t, v]): Action => ({ type: "fill", target: t, value: v })),
  fc.tuple(target, name).map(([t, v]): Action => ({ type: "select", target: t, value: v })),
  fc.tuple(fc.constantFrom("check", "uncheck") as fc.Arbitrary<"check">, target).map(([type, t]): Action => ({ type, target: t })),
);

// `expect <target>` with a text-kind HEAD is claimed by the `text` effect —
// the round-trip property quantifies over surface-REPRESENTABLE steps
// (this very property discovered the collision; see the note in ir.ts)
const visibleTarget = target.filter((t) => t.kind !== "text");

const effect: fc.Arbitrary<Effect> = fc.oneof(
  visibleTarget.map((t): Effect => ({ type: "visible", target: t })),
  target.map((t): Effect => ({ type: "absent", target: t })),
  fc.tuple(target, name, fc.boolean()).map(([t, v, exact]): Effect => ({ type: "text", target: t, value: v, exact })),
  fc.tuple(target, name).map(([t, v]): Effect => ({ type: "value", target: t, value: v })),
  target.map((t): Effect => ({ type: "checked", target: t })),
  target.map((t): Effect => ({ type: "enabled", target: t })),
  fc.tuple(target, name).map(([t, v]): Effect => ({ type: "selected", target: t, value: v })),
  path.map((p): Effect => ({ type: "url", path: p })),
  fc
    .tuple(
      fc.constantFrom("GET", "POST", "PUT", "DELETE"),
      path,
      fc.oneof(fc.constant("ok" as const), fc.integer({ min: 200, max: 599 })),
      fc.option(name, { nil: undefined }),
    )
    .map(([method, p, status, bodyContains]): Effect => {
      const e: Effect = { type: "request", method, pathPattern: p, status };
      if (bodyContains !== undefined) (e as { bodyContains?: string }).bodyContains = bodyContains;
      return e;
    }),
  fc
    .tuple(fc.constantFrom("sent", "received") as fc.Arbitrary<"sent" | "received">, name, fc.option(path, { nil: undefined }))
    .map(([dir, text, pathPattern]): Effect => {
      const e: Effect = { type: "ws", dir, text };
      if (pathPattern !== undefined) (e as { pathPattern?: string }).pathPattern = pathPattern;
      return e;
    }),
);

describe("DSL round-trip (property-based)", () => {
  it("parse(format(step)) === step for ALL representable steps", () => {
    fc.assert(
      fc.property(action, fc.array(effect, { minLength: 1, maxLength: 4 }), (a, effs) => {
        const source = `flow "roundtrip"\n${formatAction(a)}\n${effs.map((e) => `  ${formatEffect(e)}`).join("\n")}\n`;
        const flow = parseFlow(source, "prop.flow");
        expect(flow.steps).toHaveLength(1);
        expect(flow.steps[0]!.action).toEqual(a);
        expect(flow.steps[0]!.effects).toEqual(effs);
      }),
      { numRuns: 500 },
    );
  });
});
