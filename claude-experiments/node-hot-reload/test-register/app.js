import { once, defonce } from 'hot-reload/api';

const state = defonce({ count: 0 });

function getMessage() {
  return `🔥 HOT RELOADED! Count: ${state.count} 🔥`;
}

function increment() {
  state.count++;
}

// Only start once
once(setInterval(() => {
  console.log(getMessage());
}, 2000));

console.log('[app] Started! Edit this file to hot-reload.');
