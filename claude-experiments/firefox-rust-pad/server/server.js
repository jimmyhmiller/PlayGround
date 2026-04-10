const http = require("http");
const { execFile } = require("child_process");
const vm = require("vm");
const fs = require("fs");
const path = require("path");
const os = require("os");

const PORT = 7483;

function evalJavaScript(code) {
  return new Promise((resolve) => {
    try {
      let output = "";
      const sandbox = {
        console: {
          log: (...args) => { output += args.map(String).join(" ") + "\n"; },
          error: (...args) => { output += args.map(String).join(" ") + "\n"; },
          warn: (...args) => { output += args.map(String).join(" ") + "\n"; },
        },
        setTimeout, setInterval, clearTimeout, clearInterval,
        Math, Date, JSON, Array, Object, Map, Set, Promise,
        parseInt, parseFloat, isNaN, isFinite,
      };

      const result = vm.runInNewContext(code, sandbox, { timeout: 10000 });

      if (result !== undefined && output === "") {
        output = String(result) + "\n";
      }

      resolve({ success: true, output: output.trimEnd() });
    } catch (e) {
      resolve({ success: false, output: e.message });
    }
  });
}

function evalRust(code) {
  return new Promise((resolve) => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "rustpad-eval-"));
    const srcFile = path.join(tmpDir, "main.rs");
    const binFile = path.join(tmpDir, "main");

    // If code doesn't have a main function, wrap it
    const fullCode = /fn\s+main\s*\(/.test(code)
      ? code
      : `fn main() {\n${code}\n}`;

    fs.writeFileSync(srcFile, fullCode);

    execFile("rustc", [srcFile, "-o", binFile], { timeout: 30000 }, (err, stdout, stderr) => {
      if (err) {
        cleanup(tmpDir);
        resolve({ success: false, output: stderr || err.message });
        return;
      }

      execFile(binFile, [], { timeout: 10000 }, (err, stdout, stderr) => {
        cleanup(tmpDir);
        const output = (stdout + stderr).trimEnd();
        if (err && !output) {
          resolve({ success: false, output: err.message });
        } else {
          resolve({ success: !err, output: output || "(no output)" });
        }
      });
    });
  });
}

function evalClojure(code) {
  return new Promise((resolve) => {
    execFile("clojure", ["-M", "-e", code], {
      timeout: 30000,
      cwd: "/tmp",
    }, (err, stdout, stderr) => {
      const output = (stdout + stderr).trimEnd();
      if (err && !output) {
        resolve({ success: false, output: err.message });
      } else {
        resolve({ success: !err, output: output || "(no output)" });
      }
    });
  });
}

function cleanup(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch (_) {}
}

const evaluators = {
  javascript: evalJavaScript,
  rust: evalRust,
  clojure: evalClojure,
};

const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === "POST" && req.url === "/eval") {
    let body = "";
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", async () => {
      try {
        const { code, language } = JSON.parse(body);
        const evaluator = evaluators[language];

        if (!evaluator) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ success: false, output: `Unknown language: ${language}` }));
          return;
        }

        console.log(`Evaluating ${language} (${code.length} chars)`);
        const result = await evaluator(code);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ success: false, output: "Invalid request" }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(PORT, () => {
  console.log(`Rustpad eval server listening on http://localhost:${PORT}`);
  console.log("Supported languages: javascript, rust, clojure");
});
