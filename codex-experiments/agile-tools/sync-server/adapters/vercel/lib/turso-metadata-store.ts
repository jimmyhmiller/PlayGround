import { createClient, type Client } from "@libsql/client";
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

export async function createTursoMetadataStore(client: Client): Promise<MetadataStore> {
  for (const stmt of SCHEMA.split(";").filter((s) => s.trim())) {
    await client.execute(stmt);
  }

  return {
    async getAccount(accountId) {
      const result = await client.execute({
        sql: "SELECT account_id, public_key, display_name, created_at FROM accounts WHERE account_id = ?",
        args: [accountId],
      });
      if (result.rows.length === 0) return null;
      const r = result.rows[0];
      return {
        account_id: r.account_id as string,
        public_key: r.public_key as string,
        display_name: r.display_name as string,
        created_at: r.created_at as string,
      };
    },

    async createAccount(account) {
      await client.execute({
        sql: "INSERT INTO accounts (account_id, public_key, display_name, created_at) VALUES (?, ?, ?, ?)",
        args: [account.account_id, account.public_key, account.display_name, account.created_at],
      });
    },

    async getProject(ownerAccountId, projectName) {
      const result = await client.execute({
        sql: "SELECT project_id, name, account_id, created_at FROM projects WHERE account_id = ? AND name = ?",
        args: [ownerAccountId, projectName],
      });
      if (result.rows.length === 0) return null;
      const r = result.rows[0];
      return {
        project_id: r.project_id as string,
        name: r.name as string,
        account_id: r.account_id as string,
        created_at: r.created_at as string,
      };
    },

    async createProject(project) {
      await client.execute({
        sql: "INSERT INTO projects (project_id, name, account_id, created_at) VALUES (?, ?, ?, ?)",
        args: [project.project_id, project.name, project.account_id, project.created_at],
      });
    },

    async listProjectsByOwner(accountId) {
      const result = await client.execute({
        sql: "SELECT project_id, name, account_id, created_at FROM projects WHERE account_id = ?",
        args: [accountId],
      });
      return result.rows.map((r) => ({
        project_id: r.project_id as string,
        name: r.name as string,
        account_id: r.account_id as string,
        created_at: r.created_at as string,
      }));
    },

    async addMember(ownerAccountId, projectName, memberAccountId) {
      await client.execute({
        sql: "INSERT OR IGNORE INTO members (owner_account_id, project_name, member_account_id) VALUES (?, ?, ?)",
        args: [ownerAccountId, projectName, memberAccountId],
      });
    },

    async removeMember(ownerAccountId, projectName, memberAccountId) {
      await client.execute({
        sql: "DELETE FROM members WHERE owner_account_id = ? AND project_name = ? AND member_account_id = ?",
        args: [ownerAccountId, projectName, memberAccountId],
      });
    },

    async getMembers(ownerAccountId, projectName) {
      const result = await client.execute({
        sql: "SELECT member_account_id FROM members WHERE owner_account_id = ? AND project_name = ? ORDER BY added_at",
        args: [ownerAccountId, projectName],
      });
      return result.rows.map((r) => r.member_account_id as string);
    },

    async isMember(ownerAccountId, projectName, accountId) {
      const result = await client.execute({
        sql: "SELECT COUNT(*) as cnt FROM members WHERE owner_account_id = ? AND project_name = ? AND member_account_id = ?",
        args: [ownerAccountId, projectName, accountId],
      });
      return (result.rows[0]?.cnt as number ?? 0) > 0;
    },
  };
}
