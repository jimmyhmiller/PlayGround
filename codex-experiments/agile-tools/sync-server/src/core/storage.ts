export function batchPrefix(accountId: string, projectName: string): string {
  return `projects/${accountId}/${projectName}/batches/`;
}

export function batchKey(accountId: string, projectName: string, batchId: string): string {
  return `projects/${accountId}/${projectName}/batches/${batchId}.jsonl`;
}
