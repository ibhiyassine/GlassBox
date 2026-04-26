import { marked } from 'marked'

/**
 * IronClaw Markdown Parser – Uses 'marked' for full spec compliance
 * and proper rendering of headers, tables, and lists.
 */
export function renderMarkdown(text: string): string {
    if (!text) return ''

    // Basic escaping to prevent simple HTML injections
    // (though marked also handles some sanitization if configured)
    const escaped = text
        .replace(/<script/gi, '&lt;script')
        .replace(/<\/script>/gi, '&lt;/script&gt;')

    return marked.parse(escaped, {
        breaks: true,
        gfm: true
    }) as string
}
