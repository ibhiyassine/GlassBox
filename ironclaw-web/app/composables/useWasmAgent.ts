/**
 * Pyodide WASM composable – manages the Pyodide runtime lifecycle
 * and exposes tool-execution for the GlassBox ironclaw API.
 */
import { ref, readonly } from 'vue'

// Pyodide types (loaded dynamically from CDN)
declare global {
    interface Window {
        loadPyodide: (opts?: any) => Promise<any>
    }
}

const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/'

/* ────────────── Singleton state ────────────── */

const pyodide = ref<any>(null)
const isLoading = ref(false)
const isReady = ref(false)
const loadError = ref<string | null>(null)
const logs = ref<string[]>([])

let initPromise: Promise<void> | null = null

function pushLog(msg: string) {
    logs.value = [...logs.value, msg]
}

/* ────────────── Bootstrap ────────────── */

async function loadPyodideScript(): Promise<void> {
    if (typeof window.loadPyodide === 'function') return
    return new Promise((resolve, reject) => {
        const script = document.createElement('script')
        script.src = `${PYODIDE_CDN}pyodide.js`
        script.onload = () => resolve()
        script.onerror = () => reject(new Error('Failed to load Pyodide script'))
        document.head.appendChild(script)
    })
}

async function initPyodide(): Promise<void> {
    if (isReady.value || isLoading.value) return initPromise!

    isLoading.value = true
    loadError.value = null

    initPromise = (async () => {
        try {
            pushLog('📦 Loading Pyodide runtime…')
            await loadPyodideScript()

            const py = await window.loadPyodide({
                indexURL: PYODIDE_CDN,
                stdout: (text: string) => pushLog(`[py] ${text}`),
                stderr: (text: string) => pushLog(`[py:err] ${text}`),
            })
            pyodide.value = py

            pushLog('📦 Installing numpy + glassbox via micropip…')
            await py.loadPackage('micropip')
            const micropip = py.pyimport('micropip')
            await micropip.install('numpy')

            // Install glassbox from the local wheel (not on PyPI yet)
            const wheelUrl = `${window.location.origin}/whl/glassbox-0.1.0-py3-none-any.whl`
            await micropip.install(wheelUrl)

            // Pre-import ironclaw API into the Python scope
            await py.runPythonAsync(`
from glassbox.ironclaw.api import inspect_data_api, clean_data_api, train_and_tune_api
import json
print("✅ GlassBox IronClaw API loaded successfully")
`)

            isReady.value = true
            pushLog('✅ Pyodide ready – GlassBox loaded')
        } catch (err: any) {
            loadError.value = err.message ?? String(err)
            pushLog(`❌ Pyodide init failed: ${loadError.value}`)
            throw err
        } finally {
            isLoading.value = false
        }
    })()

    return initPromise
}

/* ────────────── CSV helpers ────────────── */

async function writeCsvToVFS(filename: string, csvText: string): Promise<void> {
    if (!pyodide.value) throw new Error('Pyodide not initialised')
    pyodide.value.FS.writeFile(`/home/pyodide/${filename}`, csvText)
    pushLog(`📄 Wrote ${filename} to Pyodide VFS (${csvText.length} bytes)`)
}

/* ────────────── Tool execution ────────────── */

type ToolArgs = Record<string, any>

async function callTool(name: string, args: ToolArgs): Promise<string> {
    if (!pyodide.value) throw new Error('Pyodide not initialised')

    pushLog(`🔧 Calling tool: ${name}`)

    let pythonCode: string

    switch (name) {
        case 'inspect_data': {
            // Accept either 'csv_filename' or 'filename' argument
            const filename = args.csv_filename ?? args.filename;
            if (!filename) throw new Error('inspect_data tool requires a filename argument');
            const fullPath = filename.startsWith('/') ? filename : `/home/pyodide/${filename}`;
            pythonCode = `inspect_data_api(${JSON.stringify(fullPath)})`;
            break;
        }

        case 'clean_data':
            pythonCode = `clean_data_api(
    imputation_strategy=${JSON.stringify(args.imputation_strategy)},
    outlier_handling=${JSON.stringify(args.outlier_handling)},
    encoding_strategy=${JSON.stringify(args.encoding_strategy)},
    scale=${JSON.stringify(args.scale)}
)`
            break

        case 'train_and_tune':
            pythonCode = `train_and_tune_api(
    target_column=${JSON.stringify(args.target_column)},
    models=${JSON.stringify(args.models)},
    metric=${JSON.stringify(args.metric)},
    metric_direction=${JSON.stringify(args.metric_direction)}
)`
            break

        default:
            throw new Error(`Unknown tool: ${name}`)
    }

    const wrappedCode = `
import json as _json
_result = ${pythonCode}
_result  # return value
`

    try {
        const result = await pyodide.value.runPythonAsync(wrappedCode)
        const resultStr = typeof result === 'string' ? result : String(result)
        pushLog(`✅ Tool ${name} returned ${resultStr.length} bytes`)
        return resultStr
    } catch (err: any) {
        const errorMsg = JSON.stringify({ status: 'error', message: err.message ?? String(err) })
        pushLog(`❌ Tool ${name} failed: ${err.message}`)
        return errorMsg
    }
}

/* ────────────── Composable export ────────────── */

export function useWasmAgent() {
    return {
        pyodide: readonly(pyodide),
        isLoading: readonly(isLoading),
        isReady: readonly(isReady),
        loadError: readonly(loadError),
        logs: readonly(logs),
        initPyodide,
        writeCsvToVFS,
        callTool,
    }
}
