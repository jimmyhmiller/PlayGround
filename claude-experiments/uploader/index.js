#!/usr/bin/env node

const { S3Client, PutObjectCommand, GetObjectCommand } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const fs = require("fs");
const path = require("path");

const bucket = process.env.S3_BUCKET_NAME;
const defaultExpires = parseInt(process.env.S3_SHARE_EXPIRES || "3600", 10);

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
    console.log(`Usage: s3-share <file> [expiration-seconds]

Uploads a file to S3 and returns a presigned URL.

Environment variables:
  S3_BUCKET_NAME   - S3 bucket name (required)
  S3_SHARE_EXPIRES - URL expiration in seconds (default: 3600)`);
    process.exit(args.length === 0 ? 1 : 0);
  }

  const filePath = args[0];
  const expires = args[1] ? parseInt(args[1], 10) : defaultExpires;

  if (!bucket) {
    console.error("Error: Set S3_BUCKET_NAME environment variable to your S3 bucket name");
    process.exit(1);
  }

  if (!fs.existsSync(filePath)) {
    console.error(`Error: File not found: ${filePath}`);
    process.exit(1);
  }

  const filename = path.basename(filePath);
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const s3Key = `shares/${timestamp}-${filename}`;

  const client = new S3Client({});

  console.error(`Uploading to s3://${bucket}/${s3Key}...`);

  const fileContent = fs.readFileSync(filePath);

  await client.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: s3Key,
      Body: fileContent,
    })
  );

  const url = await getSignedUrl(
    client,
    new GetObjectCommand({
      Bucket: bucket,
      Key: s3Key,
    }),
    { expiresIn: expires }
  );

  console.error(`Uploaded! URL expires in ${expires} seconds:`);
  console.log(url);
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
