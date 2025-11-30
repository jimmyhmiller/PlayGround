#!/usr/bin/env node
import * as fs from 'fs';
import * as path from 'path';

// In CommonJS module mode, __dirname is available
declare const __dirname: string;

// List of optional esbuild platforms that electron-builder might scan for
const esbuildPlatforms: string[] = [
  'aix-ppc64',
  'android-arm',
  'android-arm64',
  'android-x64',
  'darwin-arm64',
  'darwin-x64',
  'freebsd-arm64',
  'freebsd-x64',
  'linux-arm',
  'linux-arm64',
  'linux-ia32',
  'linux-loong64',
  'linux-mips64el',
  'linux-ppc64',
  'linux-riscv64',
  'linux-s390x',
  'linux-x64',
  'netbsd-arm64',
  'netbsd-x64',
  'openbsd-arm64',
  'openbsd-x64',
  'sunos-x64',
  'win32-arm64',
  'win32-ia32',
  'win32-x64'
];

const esbuildDir = path.join(__dirname, '..', 'node_modules', '@esbuild');

// Ensure @esbuild directory exists
if (!fs.existsSync(esbuildDir)) {
  fs.mkdirSync(esbuildDir, { recursive: true });
}

// Create placeholder directories for any missing platforms
let created = 0;
for (const platform of esbuildPlatforms) {
  const platformDir = path.join(esbuildDir, platform);
  if (!fs.existsSync(platformDir)) {
    fs.mkdirSync(platformDir, { recursive: true });
    created++;
  }
}

if (created > 0) {
  console.log(`Created ${created} placeholder esbuild platform directories`);
} else {
  console.log('All esbuild platform directories already exist');
}
