// Generates N feature modules (each a React component with local state + CSS) imported
// into one app, to stress the bundler's module graph at scale.
import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
const N = Number(process.argv[2] || 1500);
const root = dirname(new URL(import.meta.url).pathname);
const dir = join(root, "src/features"); mkdirSync(dir, { recursive: true });
for (let i = 0; i < N; i++) {
  writeFileSync(join(dir, `F${i}.tsx`),
`import { useState } from "react";
export default function F${i}() {
  const [n, setN] = useState(${i});
  return <div className="f"><h3>Feature ${i}</h3><button onClick={() => setN(n + 1)}>count {n}</button></div>;
}
`);
}
const imports = Array.from({ length: N }, (_, i) => `import F${i} from "./features/F${i}";`).join("\n");
const uses = Array.from({ length: N }, (_, i) => `<F${i} key={${i}} />`).join("");
writeFileSync(join(root, "src/main.tsx"),
`import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
${imports}
function App() { return (<main><h1>Scale ${N}</h1>${uses}</main>); }
createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
`);
console.log(`generated ${N} feature modules`);
