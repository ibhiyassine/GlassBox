<template>
  <div class="app">
    <!-- ── Background grain ── -->
    <div class="app__bg" />

    <!-- ── Header ── -->
    <header class="header glass animate-in">
      <div class="header__brand">
        <div class="header__logo">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="8" fill="url(#logo-grad)" />
            <path d="M10 22V10l6 6 6-6v12" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
            <defs>
              <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32">
                <stop stop-color="#6c5ce7" />
                <stop offset="1" stop-color="#a29bfe" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div>
          <h1 class="header__title">IronClaw</h1>
          <p class="header__sub">GlassBox AI Agent</p>
        </div>
      </div>

      <div class="header__status">
        <span v-if="wasm.isLoading.value" class="badge badge--loading">
          <span class="badge__dot badge__dot--pulse" /> Initializing WASM…
        </span>
        <span v-else-if="wasm.isReady.value" class="badge badge--ready">
          <span class="badge__dot badge__dot--green" /> Engine Ready
        </span>
        <span v-else class="badge badge--idle">
          <span class="badge__dot" /> Idle
        </span>
      </div>
    </header>

    <!-- ── Main layout ── -->
    <main class="main">
      <!-- Left: Upload + Logs -->
      <aside class="sidebar glass animate-in" style="animation-delay: 0.1s">
        <section class="sidebar__section">
          <h2 class="sidebar__heading">Dataset</h2>
          <CsvUpload @upload="handleCsvUpload" @remove="csvLoaded = false" />

          <div v-if="csvLoaded" class="target-col animate-in">
            <label class="target-col__label" for="target-col-input">Target column</label>
            <input
              id="target-col-input"
              v-model="targetColumn"
              type="text"
              class="target-col__input"
              placeholder="e.g. species, price…"
            />
          </div>
        </section>

        <section class="sidebar__section">
          <h2 class="sidebar__heading">Engine Logs</h2>
          <div class="logs">
            <p v-if="wasm.logs.value.length === 0" class="logs__empty">
              No activity yet.
            </p>
            <p
              v-for="(log, i) in wasm.logs.value"
              :key="i"
              class="logs__line mono"
            >
              {{ log }}
            </p>
          </div>
        </section>
      </aside>

      <!-- Right: Chat -->
      <section class="chat glass animate-in" style="animation-delay: 0.15s">
        <div ref="chatScroll" class="chat__messages">
          <!-- Welcome message -->
          <div v-if="messages.length === 0" class="chat__welcome">
            <div class="chat__welcome-icon">
              <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="var(--accent-secondary)" stroke-width="1.2">
                <circle cx="12" cy="12" r="10" />
                <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                <line x1="9" y1="9" x2="9.01" y2="9" stroke-width="2.5" />
                <line x1="15" y1="9" x2="15.01" y2="9" stroke-width="2.5" />
              </svg>
            </div>
            <h3>Welcome to IronClaw</h3>
            <p>Upload a CSV file and ask me to analyze, clean,<br/>or train models on your data.</p>
          </div>

          <!-- Messages -->
          <div
            v-for="(msg, i) in messages"
            :key="i"
            class="chat__bubble animate-in"
            :class="{
              'chat__bubble--user': msg.role === 'user',
              'chat__bubble--agent': msg.role === 'agent',
              'chat__bubble--tool': msg.role === 'tool',
            }"
          >
            <div class="chat__bubble-label">
              {{ msg.role === 'user' ? 'You' : msg.role === 'tool' ? '🔧 Tool' : '🤖 IronClaw' }}
            </div>
            <div class="chat__bubble-content" v-html="renderMarkdown(msg.text)" />
          </div>

          <!-- Typing indicator -->
          <div v-if="isThinking" class="chat__bubble chat__bubble--agent chat__typing animate-in">
            <div class="chat__bubble-label">🤖 IronClaw</div>
            <div class="chat__typing-dots">
              <span /><span /><span />
            </div>
          </div>
        </div>

        <!-- Input -->
        <form class="chat__input-bar" @submit.prevent="handleSend">
          <input
            v-model="userInput"
            type="text"
            class="chat__input"
            placeholder="Ask about your data…"
            :disabled="isThinking"
          />
          <button
            type="submit"
            class="btn btn-primary chat__send"
            :disabled="!userInput.trim() || isThinking"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import type { Content } from '@google/genai'

/* ── Composables ── */
import { useWasmAgent } from '~/composables/useWasmAgent'
const wasm = useWasmAgent()
import { useGemini } from '~/composables/useGemini'
const gemini = useGemini()

/* ── State ── */
interface ChatMessage {
  role: 'user' | 'agent' | 'tool'
  text: string
}

const messages = ref<ChatMessage[]>([])
const conversationHistory = ref<Content[]>([])
const userInput = ref('')
const isThinking = ref(false)
const csvLoaded = ref(false)
const csvFilename = ref('')
const targetColumn = ref('')
const chatScroll = ref<HTMLElement>()

/* ── CSV Upload handler ── */
async function handleCsvUpload(payload: { filename: string; csvText: string }) {
  csvFilename.value = payload.filename
  csvLoaded.value = true

  // Initialise Pyodide if not done
  if (!wasm.isReady.value) {
    await wasm.initPyodide()
  }

  // Write into Pyodide virtual FS
  await wasm.writeCsvToVFS(payload.filename, payload.csvText)
}

/* ── Markdown-ish rendering (simple) ── */
function renderMarkdown(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>')
}

/* ── Auto-scroll ── */
function scrollToBottom() {
  nextTick(() => {
    if (chatScroll.value) {
      chatScroll.value.scrollTop = chatScroll.value.scrollHeight
    }
  })
}

watch(messages, scrollToBottom, { deep: true })

/* ── Send message + agentic loop ── */
async function handleSend() {
  const text = userInput.value.trim()
  if (!text) return

  // Push user message
  messages.value.push({ role: 'user', text })
  conversationHistory.value.push({
    role: 'user',
    parts: [{ text }],
  })
  userInput.value = ''
  isThinking.value = true

  try {
    // First request to Gemini
    // Inject dataset context into the user's message for the LLM
    const context = csvLoaded.value
      ? `[Context: The user has uploaded "${csvFilename.value}"${targetColumn.value ? ` with target column "${targetColumn.value}"` : ''}. The file is loaded in the WASM filesystem.]\n\n`
      : ''
    const enrichedText = messages.value.length <= 1 ? context + text : text

    let response = await gemini.sendMessage(conversationHistory.value, enrichedText)

    // Agentic loop: keep executing tool calls until we get plain text
    const MAX_LOOPS = 10
    let loops = 0

    while (response.toolCalls.length > 0 && loops < MAX_LOOPS) {
      loops++

      // Append the assistant's function-call turn to history
      conversationHistory.value.push({
        role: 'model',
        parts: response.toolCalls.map((tc) => ({
          functionCall: { name: tc.name, args: tc.args },
        })),
      })

      // Execute each tool call via Pyodide
      const toolResults: { name: string; response: any }[] = []
      for (const tc of response.toolCalls) {
        messages.value.push({
          role: 'tool',
          text: `Calling **${tc.name}**(${JSON.stringify(tc.args).slice(0, 120)}…)`,
        })

        const result = await wasm.callTool(tc.name, tc.args)

        let parsed: any
        try {
          parsed = JSON.parse(result)
        } catch {
          parsed = { output: result }
        }

        toolResults.push({ name: tc.name, response: parsed })
      }

      // Send tool results back to Gemini
      // Add tool results as a user turn with function responses 
      conversationHistory.value.push({
        role: 'user',
        parts: toolResults.map((r) => ({
          functionResponse: { name: r.name, response: r.response },
        })),
      })

      response = await gemini.sendToolResults(
        conversationHistory.value,
        toolResults
      )
    }

    // Display the final text response
    if (response.text) {
      messages.value.push({ role: 'agent', text: response.text })
      conversationHistory.value.push({
        role: 'model',
        parts: [{ text: response.text }],
      })
    }
  } catch (err: any) {
    messages.value.push({
      role: 'agent',
      text: `❌ **Error**: ${err.message ?? String(err)}`,
    })
  } finally {
    isThinking.value = false
  }
}
</script>

<style scoped>
/* ── App shell ── */
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px 24px;
  max-width: 1440px;
  margin: 0 auto;
  position: relative;
}

.app__bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  background:
    radial-gradient(ellipse at 20% 0%, rgba(108, 92, 231, 0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 100%, rgba(162, 155, 254, 0.06) 0%, transparent 60%),
    var(--bg-primary);
}

/* ── Header ── */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
}

.header__brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.header__logo {
  flex-shrink: 0;
}

.header__title {
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header__sub {
  font-size: 0.75rem;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* ── Status badge ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 6px 14px;
  border-radius: 999px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-subtle);
}

.badge__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-muted);
}

.badge__dot--green {
  background: var(--success);
  box-shadow: 0 0 8px var(--success-glow);
}

.badge__dot--pulse {
  background: var(--accent-primary);
  animation: pulse-glow 1.4s infinite;
}

/* ── Main layout ── */
.main {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 16px;
  flex: 1;
  min-height: 0;
}

/* ── Sidebar ── */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 20px;
  overflow-y: auto;
  max-height: calc(100vh - 120px);
}

.sidebar__section {
  margin-bottom: 16px;
}

.sidebar__heading {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin-bottom: 12px;
}

/* ── Logs ── */
.logs {
  max-height: 300px;
  overflow-y: auto;
  padding: 12px;
  border-radius: var(--radius-sm);
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
}

.logs__empty {
  color: var(--text-muted);
  font-size: 0.8rem;
  text-align: center;
  padding: 16px 0;
}

.logs__line {
  font-size: 0.72rem;
  line-height: 1.7;
  color: var(--text-secondary);
  word-break: break-all;
}

/* ── Chat ── */
.chat {
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
  max-height: calc(100vh - 120px);
}

.chat__messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* Welcome */
.chat__welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 12px;
  flex: 1;
  padding: 40px 0;
}

.chat__welcome-icon {
  opacity: 0.6;
}

.chat__welcome h3 {
  font-size: 1.2rem;
  font-weight: 600;
}

.chat__welcome p {
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.6;
}

/* Bubbles */
.chat__bubble {
  max-width: 80%;
  padding: 14px 18px;
  border-radius: var(--radius-md);
  font-size: 0.88rem;
  line-height: 1.65;
}

.chat__bubble-label {
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
  color: var(--text-muted);
}

.chat__bubble--user {
  align-self: flex-end;
  background: var(--accent-gradient);
  color: #fff;
  border-bottom-right-radius: 4px;
}
.chat__bubble--user .chat__bubble-label {
  color: rgba(255, 255, 255, 0.65);
}

.chat__bubble--agent {
  align-self: flex-start;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-subtle);
  border-bottom-left-radius: 4px;
}

.chat__bubble--tool {
  align-self: flex-start;
  background: rgba(108, 92, 231, 0.08);
  border: 1px dashed var(--border-accent);
  font-family: var(--font-mono);
  font-size: 0.78rem;
  max-width: 90%;
}

.chat__bubble-content :deep(code) {
  font-family: var(--font-mono);
  background: rgba(0, 0, 0, 0.3);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.82em;
}

/* Typing dots */
.chat__typing-dots {
  display: flex;
  gap: 5px;
  padding: 4px 0;
}

.chat__typing-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-secondary);
  animation: typingBounce 1.2s infinite;
}
.chat__typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.chat__typing-dots span:nth-child(3) { animation-delay: 0.3s; }

@keyframes typingBounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}

/* Input bar */
.chat__input-bar {
  display: flex;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-subtle);
  background: var(--bg-secondary);
}

.chat__input {
  flex: 1;
  padding: 12px 18px;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 0.9rem;
  outline: none;
  transition: border-color 0.2s;
}
.chat__input::placeholder {
  color: var(--text-muted);
}
.chat__input:focus {
  border-color: var(--accent-primary);
}

.chat__send {
  padding: 12px 16px;
}

/* ── Responsive ── */
@media (max-width: 900px) {
  .main {
    grid-template-columns: 1fr;
  }
  .sidebar {
    max-height: none;
  }
  .chat {
    max-height: 60vh;
    min-height: 400px;
  }
}

/* ── Target column input ── */
.target-col {
  margin-top: 14px;
}

.target-col__label {
  display: block;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.target-col__input {
  width: 100%;
  padding: 10px 14px;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.82rem;
  outline: none;
  transition: border-color 0.2s;
}

.target-col__input::placeholder {
  color: var(--text-muted);
}

.target-col__input:focus {
  border-color: var(--accent-primary);
}
</style>
