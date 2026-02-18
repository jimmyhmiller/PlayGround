import Database from "better-sqlite3";
import type { MetadataStore, Account, Project } from "./types.js";

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

export function createSqliteMetadataStore(db: Database.Database): MetadataStore {
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.exec(SCHEMA);

  const stmts = {
    getAccount: db.prepare<[string], { account_id: string; public_key: string; display_name: string; created_at: string }>(
      "SELECT account_id, public_key, display_name, created_at FROM accounts WHERE account_id = ?"
    ),
    createAccount: db.prepare<[string, string, string, string]>(
      "INSERT INTO accounts (account_id, public_key, display_name, created_at) VALUES (?, ?, ?, ?)"
    ),
    getProject: db.prepare<[string, string], { project_id: string; name: string; account_id: string; created_at: string }>(
      "SELECT project_id, name, account_id, created_at FROM projects WHERE account_id = ? AND name = ?"
    ),
    createProject: db.prepare<[string, string, string, string]>(
      "INSERT INTO projects (project_id, name, account_id, created_at) VALUES (?, ?, ?, ?)"
    ),
    listProjectsByOwner: db.prepare<[string], { project_id: string; name: string; account_id: string; created_at: string }>(
      "SELECT project_id, name, account_id, created_at FROM projects WHERE account_id = ?"
    ),
    addMember: db.prepare<[string, string, string]>(
      "INSERT OR IGNORE INTO members (owner_account_id, project_name, member_account_id) VALUES (?, ?, ?)"
    ),
    removeMember: db.prepare<[string, string, string]>(
      "DELETE FROM members WHERE owner_account_id = ? AND project_name = ? AND member_account_id = ?"
    ),
    getMembers: db.prepare<[string, string], { member_account_id: string }>(
      "SELECT member_account_id FROM members WHERE owner_account_id = ? AND project_name = ? ORDER BY added_at"
    ),
    isMember: db.prepare<[string, string, string], { cnt: number }>(
      "SELECT COUNT(*) as cnt FROM members WHERE owner_account_id = ? AND project_name = ? AND member_account_id = ?"
    ),
  };

  return {
    async getAccount(accountId) {
      const row = stmts.getAccount.get(accountId);
      return row ?? null;
    },

    async createAccount(account) {
      stmts.createAccount.run(account.account_id, account.public_key, account.display_name, account.created_at);
    },

    async getProject(ownerAccountId, projectName) {
      const row = stmts.getProject.get(ownerAccountId, projectName);
      return row ?? null;
    },

    async createProject(project) {
      stmts.createProject.run(project.project_id, project.name, project.account_id, project.created_at);
    },

    async listProjectsByOwner(accountId) {
      return stmts.listProjectsByOwner.all(accountId);
    },

    async addMember(ownerAccountId, projectName, memberAccountId) {
      stmts.addMember.run(ownerAccountId, projectName, memberAccountId);
    },

    async removeMember(ownerAccountId, projectName, memberAccountId) {
      stmts.removeMember.run(ownerAccountId, projectName, memberAccountId);
    },

    async getMembers(ownerAccountId, projectName) {
      const rows = stmts.getMembers.all(ownerAccountId, projectName);
      return rows.map((r) => r.member_account_id);
    },

    async isMember(ownerAccountId, projectName, accountId) {
      const row = stmts.isMember.get(ownerAccountId, projectName, accountId);
      return (row?.cnt ?? 0) > 0;
    },
  };
}
