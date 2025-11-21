#!/usr/bin/env node

import { readFile, unlink, mkdtemp, readdir } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import OpenAI from 'openai';

const execPromise = promisify(exec);

// Initialize OpenAI client pointing to local qwen-vl model
const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed', // Local model doesn't need a real API key
});

/**
 * Sleep for a specified number of milliseconds
 * @param {number} ms - Milliseconds to sleep
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff
 * @param {Function} fn - Async function to retry
 * @param {number} maxRetries - Maximum number of retries (default: 3)
 * @param {number} baseDelay - Base delay in ms (default: 1000)
 * @returns {Promise<any>} Result of the function
 */
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Check if it's a connection error worth retrying
      const isRetryable =
        error.code === 'ECONNREFUSED' ||
        error.code === 'ECONNRESET' ||
        error.code === 'ETIMEDOUT' ||
        error.message?.includes('fetch failed') ||
        error.message?.includes('Connection refused') ||
        error.message?.includes('socket hang up');

      if (!isRetryable || attempt === maxRetries) {
        throw error;
      }

      // Exponential backoff: 1s, 2s, 4s, etc.
      const delay = baseDelay * Math.pow(2, attempt);
      console.error(`  Connection error (attempt ${attempt + 1}/${maxRetries + 1}): ${error.message}`);
      console.error(`  Retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Convert first N pages of PDF to images using system pdftoppm
 * @param {string} pdfPath - Path to PDF file
 * @param {number} numPages - Number of pages to convert (default: 3)
 * @param {number} dpi - Resolution in DPI (default: 72 for faster processing)
 * @returns {Promise<string[]>} Array of image file paths
 */
async function pdfToImages(pdfPath, numPages = 3, dpi = 72) {
  // Create a temporary directory for the images
  const tempDir = await mkdtemp(join(tmpdir(), 'pdf-images-'));
  const outputPrefix = join(tempDir, 'page');

  try {
    // Use system pdftoppm command to convert PDF pages to PNG
    // -png: output format
    // -f 1: first page
    // -l N: last page
    // -r DPI: resolution (lower = faster, smaller files)
    // -scale-to: max dimension in pixels (optional, controlled by DPI)
    const command = `pdftoppm -png -f 1 -l ${numPages} -r ${dpi} "${pdfPath}" "${outputPrefix}"`;

    await execPromise(command);

    // pdftoppm creates files named like: page-1.png, page-2.png, etc.
    const files = await readdir(tempDir);
    const imagePaths = files
      .filter(f => f.endsWith('.png'))
      .sort()
      .map(f => join(tempDir, f));

    return imagePaths;
  } catch (error) {
    console.error('Error converting PDF to images:', error.message);
    throw error;
  }
}

/**
 * Convert image file to base64
 * @param {string} imagePath - Path to image file
 * @returns {Promise<string>} Base64 encoded image
 */
async function imageToBase64(imagePath) {
  const imageBuffer = await readFile(imagePath);
  return imageBuffer.toString('base64');
}

/**
 * Classify document type using vision LLM
 * @param {string} pdfPath - Path to PDF file
 * @param {number} numPages - Number of pages to analyze (default: 3)
 * @param {boolean} quiet - Suppress console output (default: false)
 * @param {number} dpi - Resolution in DPI (default: 72 for faster processing)
 * @returns {Promise<{type: string, confidence: string}>}
 */
async function classifyDocumentType(pdfPath, numPages = 3, quiet = false, dpi = 72) {
  const log = quiet ? () => {} : console.log;
  const error = quiet ? () => {} : console.error;

  log(`Converting first ${numPages} pages of PDF to images (${dpi} DPI)...`);
  const imagePaths = await pdfToImages(pdfPath, numPages, dpi);

  try {
    // Convert images to base64
    log('Encoding images...');
    const imageContents = await Promise.all(
      imagePaths.map(async (path) => ({
        type: 'image_url',
        image_url: {
          url: `data:image/png;base64,${await imageToBase64(path)}`,
        },
      }))
    );

    // Prepare the message with images
    const messages = [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Please analyze these images from the first few pages of a document and classify what type of document this is.

Classify the document as one of these types:
- "paper": Academic paper, research article, or scholarly publication
- "article": Magazine article, blog post, or journalistic piece
- "book": Book, textbook, or book chapter
- "other": Anything else (slides, notes, receipts, forms, etc.)

Return ONLY a JSON object in this exact format:
{
  "type": "paper|article|book|other",
  "confidence": "high|medium|low"
}

If you cannot determine the type, use "other" with "low" confidence. Do not include any other text or explanation, only the JSON.`,
          },
          ...imageContents,
        ],
      },
    ];

    log('Sending request to qwen-vl model...');
    const response = await retryWithBackoff(async () => {
      return await client.chat.completions.create({
        model: 'qwen-vl',
        messages: messages,
        max_tokens: 500,
        temperature: 0.1,
      });
    }, 3, 1000);

    const responseText = response.choices[0].message.content.trim();
    log('Raw response:', responseText);

    // Parse the JSON response
    try {
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const classification = JSON.parse(jsonMatch[0]);
        return {
          type: classification.type || 'other',
          confidence: classification.confidence || 'low',
        };
      } else {
        error('No JSON found in response');
        return { type: 'other', confidence: 'low' };
      }
    } catch (parseError) {
      error('Error parsing JSON response:', parseError.message);
      error('Response text:', responseText);
      return { type: 'other', confidence: 'low' };
    }
  } finally {
    // Clean up temporary image files
    log('Cleaning up temporary files...');
    for (const imagePath of imagePaths) {
      try {
        await unlink(imagePath);
      } catch (error) {
        // Ignore cleanup errors
      }
    }
  }
}

/**
 * Extract title and author from PDF using vision LLM
 * @param {string} pdfPath - Path to PDF file
 * @param {number} numPages - Number of pages to analyze (default: 3)
 * @param {boolean} quiet - Suppress console output (default: false)
 * @param {number} dpi - Resolution in DPI (default: 72 for faster processing)
 * @returns {Promise<{title: string|null, author: string|null}>}
 */
async function extractMetadataWithLLM(pdfPath, numPages = 3, quiet = false, dpi = 72) {
  const log = quiet ? () => {} : console.log;
  const error = quiet ? () => {} : console.error;

  log(`Converting first ${numPages} pages of PDF to images (${dpi} DPI)...`);
  const imagePaths = await pdfToImages(pdfPath, numPages, dpi);

  try {
    // Convert images to base64
    log('Encoding images...');
    const imageContents = await Promise.all(
      imagePaths.map(async (path) => ({
        type: 'image_url',
        image_url: {
          url: `data:image/png;base64,${await imageToBase64(path)}`,
        },
      }))
    );

    // Prepare the message with images
    const messages = [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Please analyze these images from the first few pages of an academic paper or document. Extract the following information:

1. Title: The main title of the paper/document
2. Author(s): The author name(s)

Return ONLY a JSON object in this exact format:
{
  "title": "extracted title here",
  "author": "extracted author(s) here"
}

If you cannot find the title or author, use null for that field. Do not include any other text or explanation, only the JSON.`,
          },
          ...imageContents,
        ],
      },
    ];

    log('Sending request to qwen-vl model...');
    const response = await retryWithBackoff(async () => {
      return await client.chat.completions.create({
        model: 'qwen-vl', // Adjust model name if needed
        messages: messages,
        max_tokens: 500,
        temperature: 0.1, // Low temperature for more deterministic output
      });
    }, 3, 1000); // 3 retries with 1s base delay

    const responseText = response.choices[0].message.content.trim();
    log('Raw response:', responseText);

    // Parse the JSON response
    try {
      // Try to extract JSON from the response (in case there's extra text)
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const metadata = JSON.parse(jsonMatch[0]);
        return {
          title: metadata.title || null,
          author: metadata.author || null,
        };
      } else {
        error('No JSON found in response');
        return { title: null, author: null };
      }
    } catch (parseError) {
      error('Error parsing JSON response:', parseError.message);
      error('Response text:', responseText);
      return { title: null, author: null };
    }
  } finally {
    // Clean up temporary image files
    log('Cleaning up temporary files...');
    for (const imagePath of imagePaths) {
      try {
        await unlink(imagePath);
      } catch (error) {
        // Ignore cleanup errors
      }
    }
  }
}

/**
 * Main function - CLI interface
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF Metadata Extractor using Vision LLM

Usage:
  node extract-metadata.js <pdf-path> [num-pages] [--dpi N]

Arguments:
  pdf-path    Path to PDF file (required)
  num-pages   Number of pages to analyze (default: 3)

Options:
  --dpi N     Image resolution in DPI (default: 72, lower = faster)

Examples:
  node extract-metadata.js /path/to/paper.pdf
  node extract-metadata.js /path/to/paper.pdf 5
  node extract-metadata.js /path/to/paper.pdf 3 --dpi 50
  node extract-metadata.js /path/to/paper.pdf 5 --dpi 150

The tool will:
  1. Convert the first N pages of the PDF to images
  2. Send images to qwen-vl model at localhost:8080
  3. Extract title and author using vision analysis
  4. Output the results as JSON
    `);
    process.exit(0);
  }

  const pdfPath = args[0];
  const numPages = parseInt(args[1]) || 3;

  // Parse --dpi option
  const dpiIndex = args.indexOf('--dpi');
  const dpi = dpiIndex !== -1 && args[dpiIndex + 1]
    ? parseInt(args[dpiIndex + 1])
    : 72;

  try {
    console.log(`\nProcessing: ${pdfPath}`);
    console.log(`Analyzing first ${numPages} pages at ${dpi} DPI\n`);

    const metadata = await extractMetadataWithLLM(pdfPath, numPages, false, dpi);

    console.log('\n' + '='.repeat(50));
    console.log('Extracted Metadata:');
    console.log('='.repeat(50));
    console.log(JSON.stringify(metadata, null, 2));
    console.log('='.repeat(50) + '\n');
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { extractMetadataWithLLM, classifyDocumentType, pdfToImages, imageToBase64, retryWithBackoff };
