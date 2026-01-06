export declare class SemanticSearchService {
    private qdrantManager;
    private geminiService;
    private collectionName;
    constructor(collectionName?: string);
    search(params: {
        query: string;
        topK?: number;
        filters?: Record<string, any>;
        includeModuleContent?: boolean;
        minScore?: number;
        returnEmbeddings?: boolean;
    }): Promise<any[]>;
    searchInModule(query: string, moduleId: string, topK?: number): Promise<any[]>;
    searchInChapter(query: string, moduleId: string, chapterId: string, topK?: number): Promise<any[]>;
    searchWithModulePrioritization(params: {
        query: string;
        currentModule?: string;
        topK?: number;
        prioritizeCurrentModule?: boolean;
    }): Promise<any[]>;
}
export declare const getSemanticSearchService: () => SemanticSearchService;
