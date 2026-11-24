import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MAX_INPUT_BYTES = 512 * 1024;
const MAX_OUTPUT_BYTES = 256 * 1024;

async function readStdin() {
  const chunks = [];
  let size = 0;
  for await (const chunk of process.stdin) {
    size += chunk.length;
    if (size > MAX_INPUT_BYTES) throw new Error("sandbox input exceeds byte limit");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function run() {
  let payload;
  try {
    const raw = await readStdin();
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
    ({ loadPyodide } = await import(pathToFileURL(candidateModule).href));
    for (const name of ["fetch", "XMLHttpRequest", "WebSocket", "EventSource"]) {
      Object.defineProperty(globalThis, name, { value: undefined, configurable: false });
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
    const originalBinding = process.binding;
    let pyodide;
    // Emscripten NODEFS initialization needs the public fs constants on Node 20.
    process.binding = (name) => name === "constants" ? { fs: fs.constants } : originalBinding(name);
    try {
      pyodide = await loadPyodide({
        indexURL: cacheDir,
        env: { HOME: "/tmp" },
        jsglobals: Object.freeze({}),
        stdout: () => {},
        stderr: () => {},
      });
    } finally {
      process.binding = originalBinding;
    }
    const runnerPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "sandbox_runner.py");
    const runnerSource = fs.readFileSync(runnerPath, "utf8");
    pyodide.globals.set("_runner_source", runnerSource);
    pyodide.globals.set("_payload_json", JSON.stringify(payload));
    const outputJson = await pyodide.runPythonAsync(`
import json
ns = {"__name__": "_sandbox_runner"}
exec(_runner_source, ns, ns)
json.dumps(ns["execute_sandbox_payload"](json.loads(_payload_json)), ensure_ascii=True)
`);
    if (Buffer.byteLength(outputJson) > MAX_OUTPUT_BYTES) {
      throw new Error("sandbox output exceeds byte limit");
    }
    process.stdout.write(outputJson);
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
