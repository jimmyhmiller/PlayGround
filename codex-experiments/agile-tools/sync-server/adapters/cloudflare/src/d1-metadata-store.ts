import type { MetadataStore, Account, Project } from "../../../src/core/types.js";

const SCHEMA = `
CREATE TABLE IF NOT EXISTS accounts (
  account_id TEXT PRIMARY KEY,
  public_key TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  account_id TEXT NOT NULL REFERENCES accounts(account_id),
  created_at TEXT NOT NULL,
  UNIQUE(account_id, name)
);

CREATE TABLE IF NOT EXISTS members (
  owner_account_id TEXT NOT NULL,
  project_name TEXT NOT NULL,
  member_account_id TEXT NOT NULL REFERENCES accounts(account_id),
  added_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (owner_account_id, project_name, member_account_id),
  FOREIGN KEY (owner_account_id, project_name) REFERENCES projects(account_id, name)
);
`;

export async function createD1MetadataStore(db: D1Database): Promise<MetadataStore> {
  for (const stmt of SCHEMA.split(";").filter((s) => s.trim())) {
    await db.exec(stmt);
  }

  return {
    async getAccount(accountId) {
      const row = await db
        .prepare("SELECT account_id, public_key, display_name, created_at FROM accounts WHERE account_id = ?")
        .bind(accountId)
        .first<Account>();
      return row ?? null;
    },

    async createAccount(account) {
      await db
        .prepare("INSERT INTO accounts (account_id, public_key, display_name, created_at) VALUES (?, ?, ?, ?)")
        .bind(account.account_id, account.public_key, account.display_name, account.created_at)
        .run();
    },

    async getProject(ownerAccountId, projectName) {
      const row = await db
        .prepare("SELECT project_id, name, account_id, created_at FROM projects WHERE account_id = ? AND name = ?")
        .bind(ownerAccountId, projectName)
        .first<Project>();
      return row ?? null;
    },

    async createProject(project) {
      await db
        .prepare("INSERT INTO projects (project_id, name, account_id, created_at) VALUES (?, ?, ?, ?)")
        .bind(project.project_id, project.name, project.account_id, project.created_at)
        .run();
    },

    async listProjectsByOwner(accountId) {
      const result = await db
        .prepare("SELECT project_id, name, account_id, created_at FROM projects WHERE account_id = ?")
        .bind(accountId)
        .all<Project>();
      return result.results;
    },

    async addMember(ownerAccountId, projectName, memberAccountId) {
      await db
        .prepare("INSERT OR IGNORE INTO members (owner_account_id, project_name, member_account_id) VALUES (?, ?, ?)")
        .bind(ownerAccountId, projectName, memberAccountId)
        .run();
    },

    async removeMember(ownerAccountId, projectName, memberAccountId) {
      await db
        .prepare("DELETE FROM members WHERE owner_account_id = ? AND project_name = ? AND member_account_id = ?")
        .bind(ownerAccountId, projectName, memberAccountId)
        .run();
    },

    async getMembers(ownerAccountId, projectName) {
      const result = await db
        .prepare("SELECT member_account_id FROM members WHERE owner_account_id = ? AND project_name = ? ORDER BY added_at")
        .bind(ownerAccountId, projectName)
        .all<{ member_account_id: string }>();
      return result.results.map((r) => r.member_account_id);
    },

    async isMember(ownerAccountId, projectName, accountId) {
      const row = await db
        .prepare("SELECT COUNT(*) as cnt FROM members WHERE owner_account_id = ? AND project_name = ? AND member_account_id = ?")
        .bind(ownerAccountId, projectName, accountId)
        .first<{ cnt: number }>();
      return (row?.cnt ?? 0) > 0;
    },
  };
}
