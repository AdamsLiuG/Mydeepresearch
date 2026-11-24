import fs from "node:fs";
import path from "node:path";

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function run() {
  const raw = await readStdin();
  let payload;
  try {
    payload = raw.trim() ? JSON.parse(raw) : {};
  } catch (err) {
    process.stdout.write(JSON.stringify({ ok: false, error: `invalid JSON input: ${err.message}` }));
    process.exit(1);
    return;
  }

  const cacheDir = String(payload.pyodide_cache_dir || "").trim();
  if (!cacheDir || !fs.existsSync(cacheDir)) {
    process.stdout.write(
      JSON.stringify({
        ok: false,
        error: "pyodide local cache directory is unavailable",
      }),
    );
    process.exit(2);
    return;
  }

  let loadPyodide;
  try {
    const candidateModule = path.join(cacheDir, "pyodide.mjs");
    if (fs.existsSync(candidateModule)) {
      ({ loadPyodide } = await import(candidateModule));
    } else {
      ({ loadPyodide } = await import("pyodide"));
    }
  } catch (err) {
    process.stdout.write(
      JSON.stringify({
        ok: false,
        error: `failed to load pyodide runtime: ${err.message}`,
      }),
    );
    process.exit(2);
    return;
  }

  try {
    const pyodide = await loadPyodide({ indexURL: cacheDir });
    const runnerPath = path.join(path.dirname(new URL(import.meta.url).pathname), "sandbox_runner.py");
    const runnerSource = fs.readFileSync(runnerPath, "utf8");
    pyodide.globals.set("_runner_source", runnerSource);
    pyodide.globals.set("_payload_json", JSON.stringify(payload));
    const outputJson = await pyodide.runPythonAsync(`
import json
ns = {}
exec(_runner_source, ns, ns)
json.dumps(ns["execute_sandbox_payload"](json.loads(_payload_json)), ensure_ascii=False)
`);
    process.stdout.write(String(outputJson));
  } catch (err) {
    process.stdout.write(
      JSON.stringify({
        ok: false,
        error: `pyodide execution error: ${err.message}`,
      }),
    );
    process.exit(1);
  }
}

run();
