// Point the pinned cal.com checkout at the Postgres container started by
// `@calcom/prisma db-setup`, while preserving every other reproducible dummy value
// written by integration/e2e/fetch.mjs.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const app = resolve(process.argv[2] ?? "integration/e2e/.cache/calcom");
const envFile = resolve(app, ".env");
if (!existsSync(envFile)) {
  throw new Error(`missing ${envFile}; materialize cal.com with integration/e2e/fetch.mjs first`);
}

const databaseUrl = "postgresql://postgres:@127.0.0.1:5450/cal-saml";
let source = readFileSync(envFile, "utf8");

function set(name, value) {
  const line = `${name}=${JSON.stringify(value)}`;
  const pattern = new RegExp(`^${name}=.*$`, "m");
  source = pattern.test(source) ? source.replace(pattern, line) : `${source.replace(/\s*$/, "")}\n${line}\n`;
}

set("DATABASE_URL", databaseUrl);
set("DATABASE_DIRECT_URL", databaseUrl);
writeFileSync(envFile, source);
console.log(`configured ${envFile}`);
console.log(`database ${databaseUrl}`);
