<template>
  <div class="h-screen w-screen flex bg-white font-sans text-stone-800 overflow-hidden">
    
    <!-- Main Content Area -->
    <main class="flex-1 flex flex-col items-center bg-[#fdfcf9] relative overflow-hidden transition-all duration-300">
      
      <!-- Default View (Empty state) -->
      <EmptyState 
        v-if="messages.length === 0"
        v-model="userInput"
        :csv-loaded="csvLoaded"
        :csv-filename="csvFilename"
        :is-thinking="isThinking"
        :greeting="greeting"
        @upload="handleCsvUpload"
        @remove="csvLoaded = false"
        @send="handleSend"
      />

      <!-- Chat View (Active) -->
      <div v-else class="flex w-full h-full">
         <div class="flex flex-col flex-1 overflow-hidden relative">
            <header class="h-14 flex items-center justify-between px-6 border-b border-stone-100 flex-shrink-0">
               <div class="font-serif text-lg font-medium text-stone-800" style="font-family: 'Georgia', serif;">Interactive Data Analysis Chat</div>
            </header>

            <div class="flex-1 overflow-y-auto px-6 py-8 custom-scrollbar scroll-smooth" ref="chatScroll">
               <div class="max-w-3xl mx-auto flex flex-col gap-8 pb-32">
                  <ChatMessage 
                    v-for="(msg, i) in messages" 
                    :key="i"
                    :role="msg.role"
                    :text="msg.text"
                  />

                  <!-- Thinking block -->
                  <div v-if="isThinking" class="flex gap-4 w-full">
                     <div class="w-8 h-8 rounded-full bg-[#f4e7d3] text-[#cf7d41] flex items-center justify-center flex-shrink-0 mt-1">
                        <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5l-10 14M22 12H2M19 17L5 7"/></svg>
                     </div>
                     <div class="pt-2 font-medium text-stone-500 flex items-center gap-2">
                        Analyzing <span class="flex gap-1 items-center"><span class="w-1.5 h-1.5 bg-stone-300 rounded-full animate-bounce"></span><span class="w-1.5 h-1.5 bg-stone-300 rounded-full animate-bounce" style="animation-delay: 0.15s"></span><span class="w-1.5 h-1.5 bg-stone-300 rounded-full animate-bounce" style="animation-delay: 0.3s"></span></span>
                     </div>
                  </div>
               </div>
            </div>

            <!-- Sticky Bottom Input inside chat -->
            <div class="absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-3xl px-6">
               <ChatInput 
                 placeholder="Message Claude..." 
                 :disabled="isThinking"
                 @send="onChatInputSend"
               />
            </div>
         </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch, computed } from 'vue'
import { useLlm, type Content } from '~/composables/useLlm'
const llm = useLlm()
import ChatMessage from '~/components/ChatMessage.vue'
import ChatInput from '~/components/ChatInput.vue'
import EmptyState from '~/components/EmptyState.vue'

import { useWasmAgent } from '~/composables/useWasmAgent'
const wasm = useWasmAgent()

interface ChatMessageData {
  role: 'user' | 'agent' | 'tool'
  text: string
}

const messages = ref<ChatMessageData[]>([])
const conversationHistory = ref<Content[]>([])
const userInput = ref('')
const isThinking = ref(false)
const csvLoaded = ref(false)
const csvFilename = ref('')
const chatScroll = ref<HTMLElement>()

const greeting = computed(() => {
  const hour = new Date().getHours()
  if (hour < 12) return 'Morning'
  if (hour < 18) return 'Afternoon'
  return 'Evening'
})

async function handleCsvUpload(payload: { filename: string; csvText: string }) {
  csvFilename.value = payload.filename
  csvLoaded.value = true

  if (!wasm.isReady.value) {
    await wasm.initPyodide()
  }

  await wasm.writeCsvToVFS(payload.filename, payload.csvText)
}

function scrollToBottom() {
  nextTick(() => {
    if (chatScroll.value) {
      chatScroll.value.scrollTop = chatScroll.value.scrollHeight + 100
    }
  })
}

watch(messages, scrollToBottom, { deep: true })

function onChatInputSend(val: string) {
  userInput.value = val
  handleSend()
}

async function handleSend() {
  const text = userInput.value.trim()
  if (!text) return

  messages.value.push({ role: 'user', text })
  conversationHistory.value.push({
    role: 'user',
    parts: [{ text }],
  })
  userInput.value = ''
  isThinking.value = true

  try {
    const context = csvLoaded.value
      ? `[Context: The user has uploaded "${csvFilename.value}". The file is loaded in the WASM filesystem.]\n\n`
      : ''
    const enrichedText = messages.value.length <= 1 ? context + text : text

    let response = await llm.sendMessage(conversationHistory.value, enrichedText)

    const MAX_LOOPS = 10
    let loops = 0

    while (response.toolCalls.length > 0 && loops < MAX_LOOPS) {
      loops++

      conversationHistory.value.push({
        role: 'model',
        parts: response.toolCalls.map((tc: any) => ({
          functionCall: { name: tc.name, args: tc.args },
        })),
      })

      const toolResults: { name: string; response: any }[] = []
      for (const tc of response.toolCalls) {
        messages.value.push({
          role: 'tool',
          text: `Calling **${tc.name}**:\n\`\`\`json\n${JSON.stringify(tc.args, null, 2)}\n\`\`\``,
        })

        const result = await wasm.callTool(tc.name, tc.args)

        let parsed: any
        try {
          parsed = JSON.parse(result)
        } catch {
          parsed = { output: result }
        }

        // Show the execution result in the chat for transparency
        messages.value.push({
          role: 'tool',
          text: `Result from **${tc.name}**:\n\`\`\`json\n${JSON.stringify(parsed, null, 2)}\n\`\`\``,
        })

        toolResults.push({ name: tc.name, response: parsed })
      }

      conversationHistory.value.push({
        role: 'user',
        parts: toolResults.map((r) => ({
          functionResponse: { name: r.name, response: r.response },
        })),
      })

      response = await llm.sendToolResults(
        conversationHistory.value,
        toolResults
      )
    }

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

<style>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background-color: #e5e7eb;
  border-radius: 20px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background-color: #d1d5db;
}
</style>
