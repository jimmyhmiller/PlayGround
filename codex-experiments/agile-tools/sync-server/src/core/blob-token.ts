import { hmac } from "@noble/hashes/hmac";
import { sha256 } from "@noble/hashes/sha256";

const TOKEN_TTL_MS = 60 * 60 * 1000; // 1 hour

let serverSecret: Uint8Array | null = null;

export function initBlobSecret(secret?: Uint8Array): void {
  if (secret) {
    serverSecret = secret;
  } else {
    serverSecret = crypto.getRandomValues(new Uint8Array(32));
  }
}

export function getSecret(): Uint8Array {
  if (!serverSecret) {
    initBlobSecret();
  }
  return serverSecret!;
}

export function signBlobUrl(baseUrl: string, key: string): string {
  const expires = Date.now() + TOKEN_TTL_MS;
  const token = createToken(key, expires);
  return `${baseUrl}/blobs/${key}?expires=${expires}&token=${token}`;
}

export function verifyBlobToken(
  key: string,
  expires: string | null,
  token: string | null,
): boolean {
  if (!expires || !token) return false;

  const expiresNum = parseInt(expires, 10);
  if (isNaN(expiresNum)) return false;
  if (Date.now() > expiresNum) return false;

  const expected = createToken(key, expiresNum);
  return timingSafeEqual(token, expected);
}

function createToken(key: string, expires: number): string {
  const secret = getSecret();
  const message = `${key}:${expires}`;
  const mac = hmac(sha256, secret, new TextEncoder().encode(message));
  return bytesToHex(mac);
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
