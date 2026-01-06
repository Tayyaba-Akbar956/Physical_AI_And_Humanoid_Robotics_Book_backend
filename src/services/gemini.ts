import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';
import dotenv from 'dotenv';

dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
    console.warn('Warning: GEMINI_API_KEY is not set.');
}

export class GeminiService {
    private genAI: GoogleGenerativeAI;
    private embeddingModel: GenerativeModel;
    private chatModel: GenerativeModel;

    constructor() {
        this.genAI = new GoogleGenerativeAI(GEMINI_API_KEY || '');
        this.embeddingModel = this.genAI.getGenerativeModel({ model: 'embedding-001' });
        this.chatModel = this.genAI.getGenerativeModel({ model: 'gemini-pro' });
    }

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
            // The SDK might not have a direct batch embedding for strings yet, 
            // or it might be internal. For now, we'll do it sequentially or with Promise.all
            const embeddingPromises = texts.map(text => this.generateEmbedding(text));
            return await Promise.all(embeddingPromises);
        } catch (error) {
            console.error('Error in batch embedding generation:', error);
            return texts.map(() => null);
        }
    }

    async generateResponse(
        prompt: string,
        history: { role: 'user' | 'model'; parts: { text: string }[] }[] = []
    ): Promise<string> {
        if (!GEMINI_API_KEY) return 'Gemini API Key is missing.';

        try {
            const chat = this.chatModel.startChat({
                history: history,
            });

            const result = await chat.sendMessage(prompt);
            const response = await result.response;
            return response.text();
        } catch (error) {
            console.error('Error generating response with Gemini SDK:', error);
            return 'I am sorry, but I encountered an error generating a response.';
        }
    }

    getEmbeddingDimensions(): number {
        return 768; // Default for embedding-001
    }
}

let geminiServiceInstance: GeminiService | null = null;

export const getGeminiService = (): GeminiService => {
    if (!geminiServiceInstance) {
        geminiServiceInstance = new GeminiService();
    }
    return geminiServiceInstance;
};
