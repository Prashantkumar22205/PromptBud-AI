/**
 * lib/crypto.ts
 * AES-256-GCM authenticated encryption for API key storage.
 * Plaintext keys are NEVER returned to the client.
 */
import { createCipheriv, createDecipheriv, randomBytes } from "crypto";

const ALGORITHM = "aes-256-gcm";
const KEY_HEX = process.env.ENCRYPTION_KEY!;

function getKey(): Buffer {
  if (!KEY_HEX || KEY_HEX.length !== 64) {
    throw new Error("ENCRYPTION_KEY must be a 64-character hex string (32 bytes).");
  }
  return Buffer.from(KEY_HEX, "hex");
}

interface EncryptedPayload {
  ciphertext: string;
  iv: string;
  tag: string;
}

/** Encrypt a plaintext API key. Returns JSON-serialisable object. */
export function encryptApiKey(plaintext: string): string {
  const key = getKey();
  const iv = randomBytes(12); // 96-bit nonce for GCM
  const cipher = createCipheriv(ALGORITHM, key, iv);
  const encrypted = Buffer.concat([cipher.update(plaintext, "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();

  const payload: EncryptedPayload = {
    ciphertext: encrypted.toString("hex"),
    iv: iv.toString("hex"),
    tag: tag.toString("hex"),
  };
  return JSON.stringify(payload);
}

/** Decrypt a stored API key. Called only inside server-side API routes. */
export function decryptApiKey(stored: string): string {
  const key = getKey();
  const { ciphertext, iv, tag } = JSON.parse(stored) as EncryptedPayload;
  const decipher = createDecipheriv(ALGORITHM, key, Buffer.from(iv, "hex"));
  decipher.setAuthTag(Buffer.from(tag, "hex"));
  const decrypted = Buffer.concat([
    decipher.update(Buffer.from(ciphertext, "hex")),
    decipher.final(),
  ]);
  return decrypted.toString("utf8");
}

/** Returns a masked display string like ••••••••abcd */
export function maskApiKey(stored: string): string {
  try {
    const plain = decryptApiKey(stored);
    const tail = plain.slice(-4);
    return `••••••••${tail}`;
  } catch {
    return "••••••••????";
  }
}
