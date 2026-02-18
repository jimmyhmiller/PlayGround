import * as ed from "@noble/ed25519";
import { sha512 } from "@noble/hashes/sha512";
import { sha256 } from "@noble/hashes/sha256";

// ed25519 requires sha512 to be set
ed.etc.sha512Sync = (...m) => sha512(ed.etc.concatBytes(...m));

export function verifySignature(
  publicKeyB64: string,
  signature: string,
  message: string,
): boolean {
  try {
    const pubKeyBytes = base64ToBytes(publicKeyB64);
    const sigBytes = base64ToBytes(signature);
    const msgBytes = new TextEncoder().encode(message);
    return ed.verify(sigBytes, msgBytes, pubKeyBytes);
  } catch {
    return false;
  }
}

export function deriveAccountId(publicKeyB64: string): string {
  const pubKeyBytes = base64ToBytes(publicKeyB64);
  const hash = sha256(pubKeyBytes);
  const hex = bytesToHex(hash);
  return `acct_${hex.slice(0, 16)}`;
}

export function sha256Hex(data: string): string {
  const bytes = new TextEncoder().encode(data);
  const hash = sha256(bytes);
  return bytesToHex(hash);
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
