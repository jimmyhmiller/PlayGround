// The server-side dispatcher for RSC server actions.
//
// The emitted server runtime routes `POST /_action/` here. The request carries
// the action id in the `x-diffpack-action-id` header and the `encodeReply`d
// arguments as its body (a JSON string, or multipart FormData). `handleServerAction`
// decodes the arguments with `decodeReply`, resolves the real registered
// implementation through the generated action resolver (keyed by the same
// `"<moduleId>#<name>"` id the client stub was built with — the round-trip
// invariant), invokes it, and streams the result back as a flight response
// (`text/x-component`) so the client's `createFromFetch` can reconstruct it.
//
// Every failure is a hard, named error — a missing header, a missing action id,
// or a non-function export all throw rather than silently returning `undefined`.
import { decodeReply, renderToReadableStream } from "react-server-dom-webpack/server";
import { getServerActionById } from "#diffpack-rsc-action-resolver";

export async function handleServerAction(request, clientManifest = {}) {
  const id = request.headers.get("x-diffpack-action-id");
  if (!id) {
    throw new Error(
      "diffpack rsc: POST to /_action/ missing x-diffpack-action-id header",
    );
  }
  const contentType = request.headers.get("content-type") || "";
  const body = contentType.includes("multipart/form-data")
    ? await request.formData()
    : await request.text();
  const args = await decodeReply(body, {});
  const fn = await getServerActionById(id);
  const result = await fn.apply(null, args);
  const stream = renderToReadableStream(result, clientManifest);
  return new Response(stream, {
    headers: { "content-type": "text/x-component" },
  });
}

export default handleServerAction;
