#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { S3Client, PutObjectCommand, HeadObjectCommand } = require('@aws-sdk/client-s3');
const clipboard = require('clipboardy').default;
const { glob } = require('glob');




const TARGET_DIR = '/Users/jimmyhmiller/Desktop/podcast';

async function findLatestJimmyFile() {
  try {
    const files = await glob('**/*', { 
      cwd: TARGET_DIR,
      nodir: true,
      absolute: true
    });
    
    const jimmyFiles = files
      .filter(file => path.basename(file).toLowerCase().includes('jimmy'))
      .map(file => {
        const stats = fs.statSync(file);
        return {
          path: file,
          mtime: stats.mtime
        };
      });
    
    if (jimmyFiles.length === 0) {
      throw new Error('No files with "jimmy" in filename found');
    }
    
    // Sort by modification time, newest first
    jimmyFiles.sort((a, b) => b.mtime - a.mtime);
    
    return jimmyFiles[0].path;
  } catch (error) {
    throw new Error(`Error finding jimmy file: ${error.message}`);
  }
}

async function uploadToS3(filePath) {
  const s3 = new S3Client({
    region: process.env.AWS_REGION || 'us-east-1'
  });
  
  const fileName = path.basename(filePath);
  const bucketName = process.env.S3_BUCKET_NAME || 'jimmyhmiller-bucket';
  
  // Check if file already exists in S3
  const existingKey = `jimmy-uploads/${fileName}`;
  try {
    await s3.send(new HeadObjectCommand({
      Bucket: bucketName,
      Key: existingKey
    }));
    
    // File exists, return existing URL
    const existingUrl = `https://${bucketName}.s3.amazonaws.com/${existingKey}`;
    console.log('File already exists in S3, using existing URL');
    return existingUrl;
  } catch (error) {
    // File doesn't exist, proceed with upload
  }
  
  const fileContent = fs.readFileSync(filePath);
  const timestamp = Date.now();
  const key = `jimmy-uploads/${timestamp}-${fileName}`;
  
  try {
    await s3.send(new PutObjectCommand({
      Bucket: bucketName,
      Key: key,
      Body: fileContent,
      ACL: 'public-read',
      ContentType: getContentType(fileName)
    }));
    
    const publicUrl = `https://${bucketName}.s3.amazonaws.com/${key}`;
    return publicUrl;
  } catch (error) {
    throw new Error(`S3 upload failed: ${error.message}`);
  }
}

function getContentType(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  const contentTypes = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.json': 'application/json',
    '.pdf': 'application/pdf',
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.mp4': 'video/mp4',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif'
  };
  return contentTypes[ext] || 'application/octet-stream';
}

async function main() {
  try {
    console.log('Finding latest file with "jimmy" in filename...');
    const latestFile = await findLatestJimmyFile();
    console.log(`Found: ${path.basename(latestFile)}`);
    
    console.log('Uploading to S3...');
    const publicUrl = await uploadToS3(latestFile);
    
    console.log('Copying URL to clipboard...');
    clipboard.writeSync(publicUrl);
    
    console.log(`✅ Success! Public URL copied to clipboard:`);
    console.log(publicUrl);
    
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    process.exit(1);
  }
}

main();