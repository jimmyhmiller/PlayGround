// The client-side transport for RSC server actions.
//
// `rsc::transform_use_server_client` rewrites every `"use server"` export into
// `createServerReference("<moduleId>#<name>", callServer)`; React calls the
// returned action with the user's arguments and forwards them here. `callServer`
// serializes the arguments with `encodeReply` (a JSON string, or FormData when an
// argument carries a Blob/File/function/server-reference), POSTs them to the
// server action endpoint keyed by the action id, and reconstructs the flight
// response the endpoint streams back with `createFromFetch` — so an action can
// itself return client references or nested server references.
import { encodeReply, createFromFetch } from "react-server-dom-webpack/client";

export async function callServer(id, args) {
  const body = await encodeReply(args);
  const isForm = typeof body !== "string";
  const response = fetch("/_action/", {
    method: "POST",
    headers: {
      "x-diffpack-action-id": id,
      ...(isForm ? {} : { "content-type": "application/json" }),
    },
    body,
  });
  return createFromFetch(response, { callServer });
}

export default callServer;
