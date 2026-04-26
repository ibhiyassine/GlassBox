<template>
  <div class="flex-1 flex flex-col justify-center items-center w-full max-w-3xl px-6 animate-in fade-in transition-all">

    <h1 class="text-[42px] font-normal text-[#4c443b] tracking-tight mb-8 flex items-center gap-4 serif-heading" style="font-family: 'Georgia', serif;">
      {{ greeting }}
    </h1>

    <!-- Center Upload/Input Block -->
    <div class="w-full bg-white rounded-2xl shadow-[0_2px_12px_rgba(0,0,0,0.04)] border border-[#e5e1da] p-1 mb-6 transition-shadow focus-within:shadow-[0_4px_24px_rgba(0,0,0,0.06)] focus-within:border-stone-300">
      
      <div class="px-2" v-if="!csvLoaded">
         <CsvUpload @upload="$emit('upload', $event)" @remove="$emit('remove')" class="w-full mt-2" />
      </div>
      <div v-if="csvLoaded" class="flex items-center gap-2 px-4 py-2 bg-stone-50 m-2 rounded border border-stone-200">
         <span class="text-sm text-stone-600 font-medium">Dataset memory activated: {{ csvFilename }}</span>
         <button @click="$emit('remove')" class="ml-auto text-stone-400 hover:text-red-500 font-bold transition">×</button>
      </div>

      <div class="flex flex-col">
        <textarea
          :value="modelValue"
          @input="$emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
          rows="2"
          class="w-full bg-transparent px-4 py-4 outline-none text-stone-700 placeholder-stone-400 text-[15px] resize-none disabled:opacity-50 disabled:cursor-not-allowed" 
          :placeholder="csvLoaded ? 'Ask whatever you want...' : 'Upload a dataset above to chat...'"
          :disabled="!csvLoaded || isThinking"
          @keydown.enter.prevent="$emit('send')"
        ></textarea>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import CsvUpload from '~/components/CsvUpload.vue'

defineProps<{
  modelValue: string
  csvLoaded: boolean
  csvFilename: string
  isThinking: boolean
  greeting: string
}>()

defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'send'): void
  (e: 'upload', payload: any): void
  (e: 'remove'): void
}>()
</script>
