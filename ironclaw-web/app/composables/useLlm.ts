/**
 * IronClaw LLM Composable – Specialized for OpenRouter.
 * Handles model communication and tool-use translation.
 */
import { allTools } from '~/utils/tools'

/* ────────────── Types ────────────── */

export interface Part {
    text?: string
    functionCall?: { name: string; args: any }
    functionResponse?: { name: string; response: any }
}

export interface Content {
    role: 'user' | 'model'
    parts: Part[]
}

export interface ToolCall {
    name: string
    args: Record<string, any>
    error?: string | null
}

export interface LlmResponse {
    text: string | null
    toolCalls: ToolCall[]
}

/* ────────────── Composable ────────────── */

export function useLlm() {
    const config = useRuntimeConfig()

    const SYSTEM_PROMPT =
        'You are IronClaw, an expert AI data-science agent powered by the GlassBox white-box AutoML library. ' +
        'You help users inspect, clean, and train machine-learning models on their CSV data. ' +
        'Always call inspect_data first before cleaning or training. ' +
        'Explain your reasoning clearly after each step. ' +
        'When calling tools, ensure your JSON arguments are strictly valid. ' +
        'CRITICAL: Avoid common formatting errors like unclosed quotes, extra braces (e.g. `{"key": "val"}`), or repeating keys. ' +
        'The arguments must be a single, flat, valid JSON object corresponding to the tool schema.'

    function getApiKey() {
        const apiKey = config.public.openrouterApiKey as string
        if (!apiKey) throw new Error('Missing API Key in environment (NUXT_PUBLIC_OPENROUTER_API_KEY).')
        return apiKey
    }

    /**
     * Translates Gemini-style history to OpenAI-style messages for OpenRouter.
     */
    function buildMessages(history: Content[]) {
        const messages: any[] = []

        for (const entry of history) {
            const role = entry.role === 'model' ? 'assistant' : 'user'
            const parts = entry.parts || []

            // Tool responses (from the user/executor)
            const toolParts = parts.filter(p => p.functionResponse)
            if (toolParts.length > 0) {
                for (const p of toolParts) {
                    messages.push({
                        role: 'tool',
                        tool_call_id: p.functionResponse!.name,
                        content: JSON.stringify(p.functionResponse!.response)
                    })
                }
                continue
            }

            // Tool calls (from the assistant)
            const callParts = parts.filter(p => p.functionCall)
            if (callParts.length > 0) {
                messages.push({
                    role: 'assistant',
                    tool_calls: callParts.map(p => ({
                        id: p.functionCall!.name,
                        type: 'function',
                        function: {
                            name: p.functionCall!.name,
                            arguments: JSON.stringify(p.functionCall!.args)
                        }
                    }))
                })
                continue
            }

            // Text messages
            const text = parts.map(p => p.text).filter(Boolean).join('\n')
            if (text) {
                messages.push({ role, content: text })
            }
        }
        return messages
    }

    async function call(messages: any[]): Promise<LlmResponse> {
        const apiKey = getApiKey()
        let model = (config.public.openrouterModel as string) || 'google/gemini-1.5-flash'

        // OpenRouter normalization
        if (model.includes('gemini-1.5-flash') || model.includes('gemini-2.5-flash')) {
            model = 'google/gemini-1.5-flash'
        } else if (model.includes('gemini-1.5-pro')) {
            model = 'google/gemini-1.5-pro'
        } else if (!model.includes('/')) {
            // If it's a bare name like 'gemini-1.5-flash', prefix it
            model = `google/${model}`
        }

        const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'HTTP-Referer': 'https://github.com/ibhiyassine/GlassBox',
                'X-OpenRouter-Title': 'IronClaw Agent',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model,
                messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...messages],
                tools: allTools.map(t => ({
                    type: 'function',
                    function: {
                        name: t.name,
                        description: t.description,
                        parameters: t.parameters
                    }
                })),
                tool_choice: 'auto',
            })
        })

        const data = await res.json()
        if (data.error) {
            throw new Error(data.error.message || JSON.stringify(data.error))
        }

        const choice = data.choices?.[0]
        const toolCalls: ToolCall[] = []
        let text: string | null = choice?.message?.content || null

        if (choice?.message?.tool_calls) {
            for (const tc of choice.message.tool_calls) {
                if (tc.type === 'function') {
                    let args = {}
                    let parseError: string | null = null
                    try {
                        args = JSON.parse(tc.function.arguments)
                    } catch (e: any) {
                        console.error('Failed to parse tool arguments:', tc.function.arguments, e)
                        parseError = `Failed to parse tool arguments as JSON: ${e.message}. Please try again with valid JSON formatting.`
                    }
                    toolCalls.push({
                        name: tc.function.name,
                        args,
                        error: parseError // Add an error field if parsing failed
                    })
                }
            }
        }

        return { text, toolCalls }
    }

    async function sendMessage(history: Content[], userMessage: string): Promise<LlmResponse> {
        const messages = buildMessages([
            ...history,
            { role: 'user', parts: [{ text: userMessage }] }
        ])
        return call(messages)
    }

    async function sendToolResults(history: Content[], toolResults: { name: string; response: any }[]): Promise<LlmResponse> {
        const functionResponseParts: Part[] = toolResults.map((r) => ({
            functionResponse: { name: r.name, response: r.response },
        }))
        const messages = buildMessages([
            ...history,
            { role: 'user', parts: functionResponseParts }
        ])
        return call(messages)
    }

    return {
        sendMessage,
        sendToolResults,
    }
}
