import playwright from 'playwright';
import fs from 'fs';

async function captureTest(testName, url) {
    console.log(`\n=== Capturing ${testName} ===`);

    const browser = await playwright.chromium.launch({
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--disable-vulkan-surface'
        ]
    });
    const context = await browser.newContext();
    const page = await context.newPage();

    const logs = [];
    page.on('console', msg => {
        const text = msg.text();
        logs.push(text);
        console.log(`  ${text}`);
    });

    page.on('pageerror', error => {
        console.error(`  ERROR: ${error.message}`);
        logs.push(`ERROR: ${error.message}`);
    });

    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000); // Wait for rendering

    const screenshotPath = `/tmp/${testName}.png`;
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`  Screenshot saved to ${screenshotPath}`);

    // Save logs
    const logPath = `/tmp/${testName}.log`;
    fs.writeFileSync(logPath, logs.join('\n'));
    console.log(`  Logs saved to ${logPath}`);

    await browser.close();

    return { screenshotPath, logPath, logs };
}

async function main() {
    const tests = [
        ['test-storage-buffer-layout', 'http://localhost:5173/test-storage-buffer-layout.html'],
        ['test-full-quad-layout', 'http://localhost:5173/test-full-quad-layout.html'],
        ['test-incremental', 'http://localhost:5173/test-incremental.html']
    ];

    for (const [name, url] of tests) {
        try {
            await captureTest(name, url);
        } catch (error) {
            console.error(`Failed to capture ${name}:`, error.message);
        }
    }

    console.log('\n=== All tests captured ===');
}

main().catch(console.error);
