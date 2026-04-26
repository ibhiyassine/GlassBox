// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  srcDir: 'app',
  devtools: { enabled: true },

  css: ['~/assets/css/main.css'],

  // Expose .env vars to client-side code
  runtimeConfig: {
    public: {
      openrouterApiKey: '',
      openrouterModel: 'google/gemini-1.5-flash',
    },
  },

  app: {
    head: {
      title: 'IronClaw – GlassBox AI Agent',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        {
          name: 'description',
          content:
            'Client-side AI data agent powered by GlassBox and Pyodide WASM',
        },
      ],
      link: [
        {
          rel: 'preconnect',
          href: 'https://fonts.googleapis.com',
        },
        {
          rel: 'preconnect',
          href: 'https://fonts.gstatic.com',
          crossorigin: '',
        },
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap',
        },
      ],
    },
  },

  vite: {
    optimizeDeps: {
      exclude: ['pyodide'],
    },
  },

  modules: ['@nuxtjs/tailwindcss'],
})