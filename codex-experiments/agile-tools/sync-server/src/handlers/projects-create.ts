import type { HandlerRequest, HandlerResponse, MetadataStore } from "../core/types.js";
import { ProjectCreateSchema } from "../core/validate.js";
import { createProject } from "../core/projects.js";
import { ValidationError } from "../core/errors.js";

export async function handleProjectsCreate(
  req: HandlerRequest,
  store: MetadataStore,
): Promise<HandlerResponse> {
  const parsed = ProjectCreateSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const project = await createProject(store, req.accountId!, parsed.data.name);

  return {
    status: 201,
    body: JSON.stringify({ ...project, members: [] }),
  };
}
