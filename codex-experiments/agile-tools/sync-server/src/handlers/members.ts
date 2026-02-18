import type { HandlerRequest, HandlerResponse, MetadataStore } from "../core/types.js";
import { MemberAddSchema, MemberRemoveSchema, MemberListSchema } from "../core/validate.js";
import { addMember, removeMember, getMembers } from "../core/projects.js";
import { ValidationError } from "../core/errors.js";

export async function handleMemberAdd(
  req: HandlerRequest,
  store: MetadataStore,
): Promise<HandlerResponse> {
  const parsed = MemberAddSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const { project, member_account_id, owner } = parsed.data;
  const ownerAccountId = owner || req.accountId!;
  await addMember(store, req.accountId!, ownerAccountId, project, member_account_id);

  return {
    status: 200,
    body: JSON.stringify({ ok: true }),
  };
}

export async function handleMemberRemove(
  req: HandlerRequest,
  store: MetadataStore,
): Promise<HandlerResponse> {
  const parsed = MemberRemoveSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const { project, member_account_id, owner } = parsed.data;
  const ownerAccountId = owner || req.accountId!;
  await removeMember(store, req.accountId!, ownerAccountId, project, member_account_id);

  return {
    status: 200,
    body: JSON.stringify({ ok: true }),
  };
}

export async function handleMemberList(
  req: HandlerRequest,
  store: MetadataStore,
): Promise<HandlerResponse> {
  const parsed = MemberListSchema.safeParse(JSON.parse(req.body));
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }

  const { project, owner } = parsed.data;
  const ownerAccountId = owner || req.accountId!;
  const members = await getMembers(store, req.accountId!, ownerAccountId, project);

  return {
    status: 200,
    body: JSON.stringify({ members }),
  };
}
