import type { MetadataStore, Project } from "./types.js";
import { ConflictError, NotFoundError, AuthError, ValidationError } from "./errors.js";

export async function createProject(
  store: MetadataStore,
  accountId: string,
  name: string,
): Promise<Project> {
  const existing = await store.getProject(accountId, name);
  if (existing) {
    throw new ConflictError(`project '${name}' already exists.`);
  }

  const project: Project = {
    project_id: `proj_${name}_${Date.now()}`,
    name,
    account_id: accountId,
    created_at: new Date().toISOString(),
  };

  await store.createProject(project);
  return project;
}

export async function listProjects(
  store: MetadataStore,
  accountId: string,
): Promise<Project[]> {
  return store.listProjectsByOwner(accountId);
}

export async function addMember(
  store: MetadataStore,
  callerAccountId: string,
  ownerAccountId: string,
  projectName: string,
  memberAccountId: string,
): Promise<void> {
  const project = await store.getProject(ownerAccountId, projectName);
  if (!project) {
    throw new NotFoundError(`project '${projectName}' not found.`);
  }
  if (project.account_id !== callerAccountId) {
    throw new AuthError("only the project owner can add members");
  }
  if (memberAccountId === ownerAccountId) {
    return; // owner is implicitly a member
  }
  const memberAccount = await store.getAccount(memberAccountId);
  if (!memberAccount) {
    throw new ValidationError(
      `account '${memberAccountId}' does not exist. The user must sign up first.`,
    );
  }
  await store.addMember(ownerAccountId, projectName, memberAccountId);
}

export async function removeMember(
  store: MetadataStore,
  callerAccountId: string,
  ownerAccountId: string,
  projectName: string,
  memberAccountId: string,
): Promise<void> {
  const project = await store.getProject(ownerAccountId, projectName);
  if (!project) {
    throw new NotFoundError(`project '${projectName}' not found.`);
  }
  if (project.account_id !== callerAccountId) {
    throw new AuthError("only the project owner can remove members");
  }
  await store.removeMember(ownerAccountId, projectName, memberAccountId);
}

export async function getMembers(
  store: MetadataStore,
  callerAccountId: string,
  ownerAccountId: string,
  projectName: string,
): Promise<string[]> {
  const project = await store.getProject(ownerAccountId, projectName);
  if (!project) {
    throw new NotFoundError(`project '${projectName}' not found.`);
  }
  await assertAccess(store, project, callerAccountId);
  return [project.account_id, ...(await store.getMembers(ownerAccountId, projectName))];
}

export async function assertAccess(
  store: MetadataStore,
  project: Project,
  accountId: string,
): Promise<void> {
  if (project.account_id === accountId) return;
  const member = await store.isMember(project.account_id, project.name, accountId);
  if (!member) {
    throw new AuthError("you do not have access to this project");
  }
}
