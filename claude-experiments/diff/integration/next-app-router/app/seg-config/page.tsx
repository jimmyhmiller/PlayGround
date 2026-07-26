// Exercises the remaining route-segment-config exports. `runtime = "nodejs"` is the
// default diffpack serves natively; `fetchCache` / `preferredRegion` / `maxDuration`
// are advisory for a native single-node server, so diffpack parses them and emits a
// build WARN for each (never a silent default) while still rendering the page.
export const runtime = "nodejs";
export const fetchCache = "default-cache";
export const preferredRegion = "iad1";
export const maxDuration = 10;

export default function SegConfig() {
  return <main id="seg-config">seg-config page</main>;
}
