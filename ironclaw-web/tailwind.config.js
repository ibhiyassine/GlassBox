/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./app/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            },
        },
    },
    plugins: [
        require('@tailwindcss/typography'),
    ],
}
