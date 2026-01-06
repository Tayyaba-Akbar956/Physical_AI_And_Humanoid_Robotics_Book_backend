export declare class RAGAgentService {
    private geminiService;
    private searchService;
    private sessionService;
    constructor();
    getRelevantContent(params: {
        query: string;
        topK?: number;
        moduleFilter?: string;
        prioritizeCurrentModule?: boolean;
    }): Promise<any[]>;
    generateResponse(params: {
        query: string;
        context: any[];
        selectedText?: string;
        moduleContext?: string;
        conversationContext?: any[];
    }): Promise<{
        response: string;
        citations: {
            module: any;
            chapter: any;
            section: any;
            contentType: any;
            hierarchyPath: any;
        }[];
        contextUsed: number;
        responseIntent: string;
    }>;
    private analyzeResponseIntent;
}
export declare const getRAGAgentService: () => RAGAgentService;
