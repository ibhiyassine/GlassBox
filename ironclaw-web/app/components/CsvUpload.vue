<template>
  <div
    class="relative border-2 border-dashed rounded-xl transition-all duration-300 cursor-pointer overflow-hidden group"
    :class="{ 
      'border-[#D97757] bg-[#fdfcf9]': isDragOver && !file, 
      'border-stone-200 bg-stone-50 hover:border-stone-300 hover:bg-stone-100': !isDragOver && !file,
      'border-stone-200 bg-white border-solid cursor-default shadow-sm': !!file 
    }"
    @dragover.prevent="isDragOver = true"
    @dragleave.prevent="isDragOver = false"
    @drop.prevent="handleDrop"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".csv"
      class="hidden"
      @change="handleFileSelect"
    />

    <!-- Empty state -->
    <div v-if="!file" class="flex flex-col items-center gap-2 py-10 px-6" @click="($refs.fileInput as HTMLInputElement).click()">
      <div class="text-stone-400 mb-1 transition-colors duration-200 group-hover:text-stone-500">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <line x1="9" y1="15" x2="12" y2="12" />
          <line x1="15" y1="15" x2="12" y2="12" />
        </svg>
      </div>
      <p class="font-medium text-stone-700 text-sm">Drop your CSV file here</p>
      <p class="text-xs text-stone-500">or click to browse</p>
    </div>

    <!-- Loaded state -->
    <div v-else class="flex items-center gap-4 py-4 px-5 animate-in fade-in duration-300">
      <div class="flex-shrink-0 text-[#22c55e]">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <polyline points="9 15 12 18 15 15" />
        </svg>
      </div>
      <div class="flex-1 min-w-0">
        <p class="font-semibold text-stone-800 text-[14px] truncate">{{ file.name }}</p>
        <p class="text-[12px] text-stone-500 font-mono mt-0.5">
          {{ formatSize(file.size) }}
          <span v-if="rowCount"> · {{ rowCount.toLocaleString() }} rows</span>
        </p>
      </div>
      <button class="p-1.5 rounded-md text-stone-400 hover:text-red-500 hover:bg-red-50 transition-colors focus:outline-none focus:ring-2 focus:ring-red-200" @click.stop="removeFile">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  (e: 'upload', payload: { filename: string; csvText: string }): void
  (e: 'remove'): void
}>()

const fileInput = ref<HTMLInputElement>()
const isDragOver = ref(false)
const file = ref<File | null>(null)
const rowCount = ref<number | null>(null)

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

async function processFile(f: File) {
  file.value = f
  const text = await f.text()

  // Rough row count (- 1 for header)
  const lines = text.split('\n').filter((l) => l.trim().length > 0)
  rowCount.value = Math.max(0, lines.length - 1)

  emit('upload', { filename: f.name, csvText: text })
}

function handleDrop(e: DragEvent) {
  isDragOver.value = false
  const f = e.dataTransfer?.files[0]
  if (f && f.name.endsWith('.csv')) processFile(f)
}

function handleFileSelect(e: Event) {
  const target = e.target as HTMLInputElement
  const f = target.files?.[0]
  if (f) processFile(f)
}

function removeFile() {
  file.value = null
  rowCount.value = null
  if (fileInput.value) fileInput.value.value = ''
  emit('remove')
}
</script>
