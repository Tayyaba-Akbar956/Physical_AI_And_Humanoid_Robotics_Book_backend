export declare class GeminiService {
    private genAI;
    private embeddingModel;
    private chatModel;
    constructor();
    generateEmbedding(text: string): Promise<number[] | null>;
    generateEmbeddingsBatch(texts: string[]): Promise<(number[] | null)[]>;
    generateResponse(prompt: string, history?: {
        role: 'user' | 'model';
        parts: {
            text: string;
        }[];
    }[]): Promise<string>;
    getEmbeddingDimensions(): number;
}
export declare const getGeminiService: () => GeminiService;
