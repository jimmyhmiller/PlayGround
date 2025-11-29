import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
    test: {
        include: ['**/*.test.js', '**/*.e2e.test.js'],

        // Configure browser mode (enabled via BROWSER_TEST env var)
        browser: process.env.BROWSER_TEST === 'true' ? {
            provider: playwright(),
            enabled: true,
            headless: true,
            instances: [{
                browser: 'chromium',
            }],
        } : {
            enabled: false,
        },
    },
});
