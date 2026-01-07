import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';
import dotenv from 'dotenv';

dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

if (!GEMINI_API_KEY) {
    console.warn('Warning: GEMINI_API_KEY is not set. Embeddings will not work.');
}

if (!OPENROUTER_API_KEY) {
    console.warn('Warning: OPENROUTER_API_KEY is not set. Chat will not work.');
}

interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

interface OpenRouterResponse {
    choices: {
        message: {
            content: string;
        };
    }[];
}

export class GeminiService {
    private genAI: GoogleGenerativeAI;
    private embeddingModel: GenerativeModel;
    private openRouterKey: string;
    private chatModelName: string;

    constructor() {
        // Gemini for embeddings
        this.genAI = new GoogleGenerativeAI(GEMINI_API_KEY || '');
        this.embeddingModel = this.genAI.getGenerativeModel({ model: 'text-embedding-004' });

        // OpenRouter for chat
        this.openRouterKey = OPENROUTER_API_KEY || '';
        this.chatModelName = 'qwen/qwen3-coder:free';
    }

    // Uses Gemini SDK for embeddings (preserves 768 dimensions)
    async generateEmbedding(text: string): Promise<number[] | null> {
        if (!GEMINI_API_KEY) return null;

        try {
            const result = await this.embeddingModel.embedContent(text);
            const embedding = result.embedding;
            return embedding.values;
        } catch (error) {
            console.error('Error generating embedding with Gemini SDK:', error);
            return null;
        }
    }

    async generateEmbeddingsBatch(texts: string[]): Promise<(number[] | null)[]> {
        if (!GEMINI_API_KEY) return texts.map(() => null);

        try {
            const embeddingPromises = texts.map(text => this.generateEmbedding(text));
            return await Promise.all(embeddingPromises);
        } catch (error) {
            console.error('Error in batch embedding generation:', error);
            return texts.map(() => null);
        }
    }

    // Uses OpenRouter for chat (avoids Gemini rate limits)
    async generateResponse(
        prompt: string,
        history: { role: 'user' | 'model'; parts: { text: string }[] }[] = [],
        systemInstruction?: string
    ): Promise<string> {
        if (!this.openRouterKey) return 'OpenRouter API Key is missing.';

        try {
            const messages: ChatMessage[] = [];

            // Add system instruction if provided
            if (systemInstruction) {
                messages.push({
                    role: 'system',
                    content: systemInstruction,
                });
            }

            // Convert history to OpenAI-compatible format
            for (const msg of history) {
                messages.push({
                    role: msg.role === 'model' ? 'assistant' : 'user',
                    content: msg.parts.map(p => p.text).join('\n'),
                });
            }

            // Add the current user prompt
            messages.push({
                role: 'user',
                content: prompt,
            });

            const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.openRouterKey}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
                    'X-Title': 'Physical AI Book RAG',
                },
                body: JSON.stringify({
                    model: this.chatModelName,
                    messages: messages,
                    max_tokens: 2048,
                    temperature: 0.7,
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('OpenRouter chat error:', response.status, errorText);
                return 'I am sorry, but I encountered an error generating a response.';
            }

            const data: OpenRouterResponse = await response.json();
            return data.choices?.[0]?.message?.content || 'No response generated.';
        } catch (error) {
            console.error('Error generating response with OpenRouter:', error);
            return 'I am sorry, but I encountered an error generating a response.';
        }
    }

    getEmbeddingDimensions(): number {
        return 768; // Gemini text-embedding-004
    }
}

let geminiServiceInstance: GeminiService | null = null;

export const getGeminiService = (): GeminiService => {
    if (!geminiServiceInstance) {
        geminiServiceInstance = new GeminiService();
    }
    return geminiServiceInstance;
};
