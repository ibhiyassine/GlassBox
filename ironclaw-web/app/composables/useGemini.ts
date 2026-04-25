/**
 * Gemini LLM composable – wraps the @google/genai SDK for
 * multi-turn conversations with function-calling support.
 */
import { GoogleGenAI, type Content, type Part, Type } from '@google/genai'
import { allTools } from '~/utils/tools'

/* ────────────── Types ────────────── */

export interface ToolCall {
    name: string
    args: Record<string, any>
}

export interface GeminiResponse {
    text: string | null
    toolCalls: ToolCall[]
}

/* ────────────── Composable ────────────── */

export function useGemini() {
    const config = useRuntimeConfig()

    function getClient() {
        const apiKey = config.public.geminiApiKey as string
        if (!apiKey) {
            throw new Error(
                'Missing NUXT_PUBLIC_GEMINI_API_KEY. Set it in your .env file.'
            )
        }
        return new GoogleGenAI({ apiKey })
    }

    /**
     * Sends a multi-turn conversation to Gemini with the GlassBox tool schemas.
     * Returns either a text reply or a list of tool calls to execute.
     */
    async function sendMessage(
        history: Content[],
        userMessage: string
    ): Promise<GeminiResponse> {
        const client = getClient()
        const model = (config.public.geminiModel as string) || 'gemini-2.5-flash'

        const contents: Content[] = [
            ...history,
            { role: 'user', parts: [{ text: userMessage }] },
        ]

        const response = await client.models.generateContent({
            model,
            contents,
            config: {
                tools: [{ functionDeclarations: allTools }],
                systemInstruction:
                    'You are IronClaw, an expert AI data-science agent powered by the GlassBox white-box AutoML library. ' +
                    'You help users inspect, clean, and train machine-learning models on their CSV data. ' +
                    'Always call inspect_data first before cleaning or training. ' +
                    'Explain your reasoning clearly after each step. ' +
                    'When you receive tool results, interpret them for the user in plain language with actionable insights.',
            },
        })

        const candidate = response.candidates?.[0]
        if (!candidate?.content?.parts) {
            return { text: 'No response from the model.', toolCalls: [] }
        }

        const toolCalls: ToolCall[] = []
        let text: string | null = null

        for (const part of candidate.content.parts) {
            if (part.functionCall) {
                toolCalls.push({
                    name: part.functionCall.name!,
                    args: (part.functionCall.args as Record<string, any>) ?? {},
                })
            }
            if (part.text) {
                text = (text ?? '') + part.text
            }
        }

        return { text, toolCalls }
    }

    /**
     * Continue the conversation after executing tool calls,
     * sending the function results back to Gemini.
     */
    async function sendToolResults(
        history: Content[],
        toolResults: { name: string; response: any }[]
    ): Promise<GeminiResponse> {
        const client = getClient()
        const model = (config.public.geminiModel as string) || 'gemini-2.5-flash'

        const functionResponseParts: Part[] = toolResults.map((r) => ({
            functionResponse: {
                name: r.name,
                response: r.response,
            },
        }))

        const contents: Content[] = [
            ...history,
            { role: 'user', parts: functionResponseParts },
        ]

        const response = await client.models.generateContent({
            model,
            contents,
            config: {
                tools: [{ functionDeclarations: allTools }],
                systemInstruction:
                    'You are IronClaw, an expert AI data-science agent powered by the GlassBox white-box AutoML library. ' +
                    'You help users inspect, clean, and train machine-learning models on their CSV data. ' +
                    'Always call inspect_data first before cleaning or training. ' +
                    'Explain your reasoning clearly after each step. ' +
                    'When you receive tool results, interpret them for the user in plain language with actionable insights.',
            },
        })

        const candidate = response.candidates?.[0]
        if (!candidate?.content?.parts) {
            return { text: 'No response from the model.', toolCalls: [] }
        }

        const toolCalls: ToolCall[] = []
        let text: string | null = null

        for (const part of candidate.content.parts) {
            if (part.functionCall) {
                toolCalls.push({
                    name: part.functionCall.name!,
                    args: (part.functionCall.args as Record<string, any>) ?? {},
                })
            }
            if (part.text) {
                text = (text ?? '') + part.text
            }
        }

        return { text, toolCalls }
    }

    return {
        sendMessage,
        sendToolResults,
    }
}
