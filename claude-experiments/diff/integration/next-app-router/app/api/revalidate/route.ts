// A Route Handler that performs on-demand revalidation (next/cache). `POST
// /api/revalidate?tag=products` busts every prerendered page tagged `products` (the
// /products page); `POST /api/revalidate?path=/isr` busts that exact prerendered page.
// diffpack collects the revalidatePath/revalidateTag calls off the handler's per-request
// store and the orchestrator marks the matching cache entries stale, so the next request
// serves STALE and regenerates in the background. `next build`/`next start` accept this
// unchanged (revalidatePath/revalidateTag are real next/cache exports usable in a handler).
import { revalidatePath, revalidateTag } from "next/cache";

export async function POST(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const tag = url.searchParams.get("tag");
  const path = url.searchParams.get("path");
  if (tag) revalidateTag(tag, "max");
  if (path) revalidatePath(path);
  return new Response(
    JSON.stringify({ revalidated: true, tag: tag ?? null, path: path ?? null }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}
