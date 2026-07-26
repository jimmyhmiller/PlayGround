// Parallel-slot default: rendered for the @team slot when no @team sub-route matches
// (Next requires a default.js for every parallel slot). Hard navigation / refresh use it.
export default function TeamDefault() {
  return null;
}
