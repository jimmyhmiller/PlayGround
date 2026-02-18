import Database from "better-sqlite3";
import type { MetadataStore, BlobStore } from "../../src/core/types.js";
import { createSqliteMetadataStore } from "../../src/core/sqlite-metadata-store.js";

export function createTestMetadataStore(): MetadataStore {
  const db = new Database(":memory:");
  return createSqliteMetadataStore(db);
}

// Mock blob store that stores blobs in memory and returns fake URLs
// The "URLs" are just `mock://blob/{key}` — tests interact with the store directly
export function createMockBlobStore(): BlobStore & {
  // Expose internals for tests to upload/download blobs directly
  putBlob(key: string, value: string): void;
  getBlob(key: string): string | null;
} {
  const blobs = new Map<string, string>();

  return {
    async createUploadUrl(key: string): Promise<string> {
      return `mock://blob/${key}`;
    },
    async createDownloadUrl(key: string): Promise<string> {
      return `mock://blob/${key}`;
    },
    async listKeys(prefix: string): Promise<string[]> {
      const keys: string[] = [];
      for (const key of blobs.keys()) {
        if (key.startsWith(prefix)) {
          keys.push(key);
        }
      }
      return keys.sort();
    },
    putBlob(key: string, value: string): void {
      blobs.set(key, value);
    },
    getBlob(key: string): string | null {
      return blobs.get(key) ?? null;
    },
  };
}
