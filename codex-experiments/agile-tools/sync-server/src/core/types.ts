export interface HandlerRequest {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: string;
  accountId?: string;
}

export interface HandlerResponse {
  status: number;
  body: string;
  headers?: Record<string, string>;
}

export type Handler = (req: HandlerRequest) => Promise<HandlerResponse>;

// Blob storage — generates URLs for clients to upload/download batch files directly
export interface BlobStore {
  createUploadUrl(key: string): Promise<string>;
  createDownloadUrl(key: string): Promise<string>;
  listKeys(prefix: string): Promise<string[]>;
}

// Typed metadata store — replaces the old key-value StorageClient
export interface MetadataStore {
  getAccount(accountId: string): Promise<Account | null>;
  createAccount(account: Account): Promise<void>;

  getProject(ownerAccountId: string, projectName: string): Promise<Project | null>;
  createProject(project: Project): Promise<void>;
  listProjectsByOwner(accountId: string): Promise<Project[]>;

  addMember(ownerAccountId: string, projectName: string, memberAccountId: string): Promise<void>;
  removeMember(ownerAccountId: string, projectName: string, memberAccountId: string): Promise<void>;
  getMembers(ownerAccountId: string, projectName: string): Promise<string[]>;
  isMember(ownerAccountId: string, projectName: string, accountId: string): Promise<boolean>;
}

export interface Account {
  account_id: string;
  public_key: string;
  display_name: string;
  created_at: string;
}

export interface Project {
  project_id: string;
  name: string;
  account_id: string;
  created_at: string;
}
