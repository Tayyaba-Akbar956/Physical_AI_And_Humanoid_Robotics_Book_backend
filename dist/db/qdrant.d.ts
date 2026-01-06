export interface QdrantPoint {
    id: string;
    vector: number[];
    payload: Record<string, any>;
}
export declare class QdrantManager {
    private client;
    private collectionName;
    constructor(collectionName?: string);
    createCollectionIfNotExists(vectorSize?: number): Promise<void>;
    addEmbeddings(points: QdrantPoint[]): Promise<boolean>;
    searchSimilar(queryVector: number[], topK?: number, filters?: Record<string, any>): Promise<any[]>;
    getEmbeddingById(embeddingId: string): Promise<any | null>;
    deleteEmbedding(embeddingId: string): Promise<boolean>;
    getCollectionInfo(): Promise<any>;
}
export declare const getQdrantManager: () => QdrantManager;
