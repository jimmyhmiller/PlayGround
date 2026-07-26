// nav-demo/alpha: a Server Component. useSelectedLayoutSegment() in the parent layout
// resolves to "alpha" here. It renders a changing server value (Date.now) inside a
// client RefreshControls island (so router.refresh() re-fetches this server flight while
// the island's local count survives), plus a next/dynamic(ssr:false) host.
import RefreshControls from "./RefreshControls";
import DynamicHost from "./DynamicHost";

export const dynamic = "force-dynamic";

export default function Alpha() {
  const now = Date.now();
  return (
    <div id="alpha">
      <RefreshControls>
        <span id="server-now">server-now: {now}</span>
      </RefreshControls>
      <DynamicHost />
    </div>
  );
}
