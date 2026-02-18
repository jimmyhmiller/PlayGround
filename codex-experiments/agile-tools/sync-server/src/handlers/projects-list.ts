import type { HandlerRequest, HandlerResponse, MetadataStore } from "../core/types.js";
import { listProjects } from "../core/projects.js";

export async function handleProjectsList(
  req: HandlerRequest,
  store: MetadataStore,
): Promise<HandlerResponse> {
  const projects = await listProjects(store, req.accountId!);

  return {
    status: 200,
    body: JSON.stringify({ projects }),
  };
}
