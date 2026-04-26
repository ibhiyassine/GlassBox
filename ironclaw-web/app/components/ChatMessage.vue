<template>
  <div class="flex gap-4 w-full">
    <!-- Avatar User -->
    <div v-if="role === 'user'" class="w-8 h-8 rounded-full bg-stone-800 text-white flex items-center justify-center flex-shrink-0 font-semibold text-xs mt-1">AY</div>
    <!-- Avatar System/Agent -->
    <div v-else-if="role === 'agent' || role === 'tool'" class="w-8 h-8 rounded-full bg-[#f4e7d3] text-[#cf7d41] flex items-center justify-center flex-shrink-0 mt-1">
      <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5l-10 14M22 12H2M19 17L5 7"/></svg>
    </div>

    <!-- Message Body -->
    <div class="flex-1 max-w-[100%] overflow-hidden pt-1.5">
      <!-- Tool message custom box -->
      <div v-if="role === 'tool'" class="bg-stone-50 border border-stone-200 rounded-lg p-4 font-mono text-sm text-stone-600 overflow-x-auto shadow-sm">
         <div v-html="renderedText"></div>
      </div>
      <!-- Normal message -->
      <div v-else class="prose prose-stone max-w-none prose-p:leading-relaxed prose-pre:bg-stone-50 prose-pre:border prose-pre:border-stone-200 prose-pre:text-stone-800 prose-code:text-[#D97757] prose-headings:font-serif prose-headings:font-normal prose-a:text-blue-600 prose-h2:text-2xl md-content" v-html="renderedText"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { renderMarkdown } from '~/utils/markdown'

const props = defineProps<{
  role: 'user' | 'agent' | 'tool'
  text: string
}>()

const renderedText = computed(() => renderMarkdown(props.text))
</script>

<style>
.md-content ul { list-style-type: disc; margin-left: 1.5rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
.md-content ol { list-style-type: decimal; margin-left: 1.5rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
</style>
