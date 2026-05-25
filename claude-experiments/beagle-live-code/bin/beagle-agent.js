#!/usr/bin/env node
const { spawn } = require("child_process");
const path = require("path");

const projectDir = path.resolve(__dirname, "..");
const tsxBin = path.join(projectDir, "node_modules", ".bin", "tsx");
const agentFile = path.join(projectDir, "beagle_agent.ts");

const child = spawn(tsxBin, [agentFile, ...process.argv.slice(2)], {
  stdio: "inherit",
  cwd: process.cwd(),
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 0);
});
