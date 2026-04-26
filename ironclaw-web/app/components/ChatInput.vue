<template>
  <div class="w-full">
    <form @submit.prevent="handleSubmit" class="bg-white rounded-2xl shadow-[0_4px_24px_rgba(0,0,0,0.08)] border border-stone-200 p-1 flex flex-col focus-within:shadow-[0_8px_32px_rgba(0,0,0,0.12)] transition-shadow">
      <textarea
        v-model="internalValue"
        rows="1"
        class="w-full bg-transparent px-4 py-3 outline-none text-stone-700 placeholder-stone-400 text-[15px] resize-none disabled:opacity-50" 
        :placeholder="placeholder"
        :disabled="disabled"
        @keydown.enter.prevent="handleSubmit"
      ></textarea>
      
      <div class="flex justify-end items-center px-4 pb-2 pt-1 border-stone-100">
        <button type="submit" :disabled="!internalValue.trim() || disabled" class="w-8 h-8 rounded-full bg-stone-800 text-white hover:bg-stone-700 disabled:opacity-40 disabled:hover:bg-stone-800 transition flex items-center justify-center">
          <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 19V5m0 0l-7 7m7-7l7 7" /></svg>
        </button>
      </div>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  placeholder?: string
  disabled?: boolean
}>()

const emit = defineEmits<{
  (e: 'send', value: string): void
}>()

const internalValue = ref('')

function handleSubmit() {
  const val = internalValue.value.trim()
  if (val && !props.disabled) {
    emit('send', val)
    internalValue.value = ''
  }
}
</script>
