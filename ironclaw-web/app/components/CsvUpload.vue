<template>
  <div
    class="csv-upload"
    :class="{ 'csv-upload--active': isDragOver, 'csv-upload--loaded': !!file }"
    @dragover.prevent="isDragOver = true"
    @dragleave.prevent="isDragOver = false"
    @drop.prevent="handleDrop"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".csv"
      class="csv-upload__input"
      @change="handleFileSelect"
    />

    <!-- Empty state -->
    <div v-if="!file" class="csv-upload__prompt" @click="($refs.fileInput as HTMLInputElement).click()">
      <div class="csv-upload__icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <line x1="9" y1="15" x2="12" y2="12" />
          <line x1="15" y1="15" x2="12" y2="12" />
        </svg>
      </div>
      <p class="csv-upload__title">Drop your CSV file here</p>
      <p class="csv-upload__subtitle">or click to browse</p>
    </div>

    <!-- Loaded state -->
    <div v-else class="csv-upload__loaded animate-in">
      <div class="csv-upload__file-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <polyline points="9 15 12 18 15 15" stroke="var(--success)" />
        </svg>
      </div>
      <div class="csv-upload__meta">
        <p class="csv-upload__filename">{{ file.name }}</p>
        <p class="csv-upload__size">
          {{ formatSize(file.size) }}
          <span v-if="rowCount"> · {{ rowCount.toLocaleString() }} rows</span>
        </p>
      </div>
      <button class="btn btn-ghost csv-upload__remove" @click.stop="removeFile">
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

<style scoped>
.csv-upload {
  position: relative;
  border: 2px dashed var(--border-subtle);
  border-radius: var(--radius-md);
  transition: all 0.3s ease;
  cursor: pointer;
}

.csv-upload:hover,
.csv-upload--active {
  border-color: var(--accent-primary);
  background: var(--accent-glow);
}

.csv-upload--loaded {
  border-style: solid;
  border-color: var(--success);
  background: var(--success-glow);
  cursor: default;
}

.csv-upload__input {
  display: none;
}

.csv-upload__prompt {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 40px 24px;
}

.csv-upload__icon {
  color: var(--text-muted);
  margin-bottom: 4px;
  transition: color 0.2s;
}
.csv-upload:hover .csv-upload__icon {
  color: var(--accent-secondary);
}

.csv-upload__title {
  font-weight: 500;
  font-size: 0.95rem;
  color: var(--text-primary);
}

.csv-upload__subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.csv-upload__loaded {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 20px;
}

.csv-upload__file-icon {
  flex-shrink: 0;
}

.csv-upload__meta {
  flex: 1;
  min-width: 0;
}

.csv-upload__filename {
  font-weight: 600;
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.csv-upload__size {
  font-size: 0.78rem;
  color: var(--text-secondary);
  font-family: var(--font-mono);
}

.csv-upload__remove {
  padding: 6px;
  flex-shrink: 0;
}
</style>
