import { QdrantClient } from '@qdrant/js-client-rest';
import dotenv from 'dotenv';
dotenv.config();
const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const QDRANT_API_KEY = process.env.QDRANT_API_KEY || 'dummy';
export class QdrantManager {
    client;
    collectionName;
    constructor(collectionName = 'textbook_content_embeddings') {
        this.collectionName = collectionName;
        this.client = new QdrantClient({
            url: QDRANT_URL,
            apiKey: QDRANT_API_KEY,
        });
    }
    async createCollectionIfNotExists(vectorSize = 768) {
        try {
            const collections = await this.client.getCollections();
            const exists = collections.collections.some(c => c.name === this.collectionName);
            if (!exists) {
                console.log(`Creating collection '${this.collectionName}'...`);
                await this.client.createCollection(this.collectionName, {
                    vectors: {
                        size: vectorSize,
                        distance: 'Cosine',
                    },
                });
                console.log(`Collection '${this.collectionName}' created successfully`);
            }
            else {
                console.log(`Collection '${this.collectionName}' already exists`);
            }
        }
        catch (error) {
            console.error(`Error creating/checking collection:`, error);
            throw error;
        }
    }
    async addEmbeddings(points) {
        try {
            await this.client.upsert(this.collectionName, {
                wait: true,
                points: points.map(p => ({
                    id: p.id,
                    vector: p.vector,
                    payload: p.payload,
                })),
            });
            return true;
        }
        catch (error) {
            console.error('Error adding embeddings to Qdrant:', error);
            return false;
        }
    }
    async searchSimilar(queryVector, topK = 5, filters) {
        try {
            let qdrantFilter = undefined;
            if (filters && Object.keys(filters).length > 0) {
                qdrantFilter = {
                    must: Object.entries(filters).map(([key, value]) => ({
                        key,
                        match: { value },
                    })),
                };
            }
            const results = await this.client.search(this.collectionName, {
                vector: queryVector,
                limit: topK,
                filter: qdrantFilter,
            });
            return results.map(hit => ({
                id: hit.id,
                payload: hit.payload,
                score: hit.score,
            }));
        }
        catch (error) {
            console.error('Error searching in Qdrant:', error);
            return [];
        }
    }
    async getEmbeddingById(embeddingId) {
        try {
            const records = await this.client.retrieve(this.collectionName, {
                ids: [embeddingId],
                with_payload: true,
                with_vector: true,
            });
            if (records.length > 0) {
                const record = records[0];
                return {
                    id: record.id,
                    vector: record.vector,
                    payload: record.payload,
                };
            }
            return null;
        }
        catch (error) {
            console.error('Error retrieving embedding from Qdrant:', error);
            return null;
        }
    }
    async deleteEmbedding(embeddingId) {
        try {
            await this.client.delete(this.collectionName, {
                points: [embeddingId],
            });
            return true;
        }
        catch (error) {
            console.error('Error deleting embedding from Qdrant:', error);
            return false;
        }
    }
    async getCollectionInfo() {
        try {
            const collectionInfo = await this.client.getCollection(this.collectionName);
            return {
                collection_name: this.collectionName,
                vector_size: collectionInfo.config.params.vectors.size,
                distance: collectionInfo.config.params.vectors.distance,
            };
        }
        catch (error) {
            console.error('Error getting collection info:', error);
            return null;
        }
    }
}
let qdrantManagerInstance = null;
export const getQdrantManager = () => {
    if (!qdrantManagerInstance) {
        qdrantManagerInstance = new QdrantManager();
    }
    return qdrantManagerInstance;
};
//# sourceMappingURL=qdrant.js.map