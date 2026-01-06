import { getQdrantManager } from '../db/qdrant.js';
import { getGeminiService } from './gemini.js';
export class SemanticSearchService {
    qdrantManager;
    geminiService;
    collectionName;
    constructor(collectionName = 'textbook_content_embeddings') {
        this.collectionName = collectionName;
        this.qdrantManager = getQdrantManager();
        this.geminiService = getGeminiService();
    }
    async search(params) {
        const { query, topK = 5, filters, includeModuleContent = true, minScore = 0.3, returnEmbeddings = false } = params;
        try {
            const queryEmbedding = await this.geminiService.generateEmbedding(query);
            if (!queryEmbedding) {
                console.error('Failed to generate embedding for query');
                return [];
            }
            const searchResults = await this.qdrantManager.searchSimilar(queryEmbedding, topK, filters);
            const filteredResults = searchResults.filter(r => r.score >= minScore);
            const formattedResults = filteredResults.map(result => {
                const payload = result.payload;
                const formatted = {
                    id: result.id,
                    score: result.score,
                    moduleId: payload.module_id || '',
                    chapterId: payload.chapter_id || '',
                    sectionId: payload.section_id || '',
                    hierarchyPath: payload.hierarchy_path || '',
                    contentType: payload.content_type || '',
                    metadata: payload.metadata || {},
                    similarityScore: result.score,
                };
                if (includeModuleContent) {
                    formatted.content = payload.content || '';
                }
                if (returnEmbeddings) {
                    formatted.embedding = result.vector || [];
                }
                return formatted;
            });
            return formattedResults.sort((a, b) => b.score - a.score);
        }
        catch (error) {
            console.error('Error in semantic search:', error);
            return [];
        }
    }
    async searchInModule(query, moduleId, topK = 5) {
        return this.search({ query, topK, filters: { module_id: moduleId } });
    }
    async searchInChapter(query, moduleId, chapterId, topK = 5) {
        return this.search({
            query,
            topK,
            filters: {
                module_id: moduleId,
                chapter_id: chapterId,
            },
        });
    }
    async searchWithModulePrioritization(params) {
        const { query, currentModule, topK = 5, prioritizeCurrentModule = true } = params;
        if (!prioritizeCurrentModule || !currentModule) {
            return this.search({ query, topK });
        }
        try {
            const currentModuleResults = await this.searchInModule(query, currentModule, topK);
            if (currentModuleResults.length >= topK) {
                return currentModuleResults.slice(0, topK);
            }
            const additionalNeeded = topK - currentModuleResults.length;
            const allResults = await this.search({ query, topK: topK * 2 });
            const otherModuleResults = allResults.filter(r => r.moduleId !== currentModule);
            const combinedResults = [...currentModuleResults, ...otherModuleResults.slice(0, additionalNeeded)];
            return combinedResults.sort((a, b) => b.score - a.score).slice(0, topK);
        }
        catch (error) {
            console.error('Error in search with module prioritization:', error);
            return this.search({ query, topK });
        }
    }
}
let semanticSearchServiceInstance = null;
export const getSemanticSearchService = () => {
    if (!semanticSearchServiceInstance) {
        semanticSearchServiceInstance = new SemanticSearchService();
    }
    return semanticSearchServiceInstance;
};
//# sourceMappingURL=semantic-search.js.map