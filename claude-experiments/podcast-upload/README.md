# Podcast Upload Utility

A Node.js utility that finds the latest file with "jimmy" in the filename in `/Users/jimmyhmiller/Desktop/podcast` and uploads it to S3 with public access, copying the URL to your clipboard.

## Installation

```bash
cd podcast-upload
npm install
npm install -g .
```

## Setup

Set these environment variables:

```bash
export S3_BUCKET_NAME=jimmyhmiller-bucket  # optional, defaults to jimmyhmiller-bucket
export AWS_REGION=us-east-1  # optional, defaults to us-east-1
```

Make sure your AWS credentials are configured (via AWS CLI, environment variables, or IAM role).

## Usage

```bash
podcast-upload
```

The utility will:
1. Search for files with "jimmy" in filename in `/Users/jimmyhmiller/Desktop/podcast`
2. Find the most recently modified one
3. Upload it to S3 with public read access
4. Copy the public URL to your clipboard

## Requirements

- Node.js
- AWS credentials configured
- S3 bucket with public read permissions enabled