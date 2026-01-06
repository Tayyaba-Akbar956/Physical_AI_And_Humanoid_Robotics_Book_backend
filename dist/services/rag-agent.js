import { getGeminiService } from './gemini.js';
import { getSemanticSearchService } from './semantic-search.js';
import { getSessionManagementService } from './session-manager.js';
export class RAGAgentService {
    geminiService;
    searchService;
    sessionService;
    constructor() {
        this.geminiService = getGeminiService();
        this.searchService = getSemanticSearchService();
        this.sessionService = getSessionManagementService();
    }
    async getRelevantContent(params) {
        const { query, topK = 5, moduleFilter, prioritizeCurrentModule = true } = params;
        return this.searchService.searchWithModulePrioritization({
            query,
            currentModule: moduleFilter,
            topK,
            prioritizeCurrentModule,
        });
    }
    async generateResponse(params) {
        const { query, context, selectedText, moduleContext, conversationContext = [] } = params;
        const contextText = context
            .map((chunk, i) => `Relevant Content ${i + 1} (Module: ${chunk.moduleId}, Chapter: ${chunk.chapterId}):\n${chunk.content}\n`)
            .join('\n');
        const history = conversationContext.map(msg => ({
            role: msg.senderType === 'ai_agent' ? 'model' : 'user',
            parts: [{ text: msg.content }],
        }));
        const citations = context.map(chunk => ({
            module: chunk.moduleId,
            chapter: chunk.chapterId,
            section: chunk.sectionId,
            contentType: chunk.contentType,
            hierarchyPath: chunk.hierarchyPath,
        }));
        const intent = this.analyzeResponseIntent(query);
        let systemPrompt = `You are an educational assistant helping students with the Physical AI & Humanoid Robotics textbook.
Only use information from the provided textbook content. Cite which modules/chapters the information comes from using the format: "According to Module X, Chapter Y...".
If the answer isn't in the textbook, clearly state this. Use textbook terminology.
Keep response between 150-300 words. Format with clear paragraphs and bullet points if appropriate.

Textbook Content:
${contextText}

Contextual Intent: ${intent}
`;
        if (selectedText) {
            systemPrompt += `\nThe student has selected the following text: "${selectedText}"\nPrioritize explaining the selected text but enrich with related content if helpful.`;
        }
        if (moduleContext) {
            systemPrompt += `\nThe student is currently studying Module: ${moduleContext}. Prioritize responses using content from current module when possible.`;
        }
        const responseText = await this.geminiService.generateResponse(query, history);
        return {
            response: responseText,
            citations,
            contextUsed: context.length,
            responseIntent: intent,
        };
    }
    analyzeResponseIntent(query) {
        const queryLower = query.toLowerCase();
        if (/explain|describe|what is|how does|why is/.test(queryLower))
            return 'explanation';
        if (/summarize|summary|overview|briefly/.test(queryLower))
            return 'summary';
        if (/example|example of|show me|like/.test(queryLower))
            return 'example';
        if (/compare|difference|vs|versus/.test(queryLower))
            return 'comparison';
        if (/definition|define|meaning/.test(queryLower))
            return 'definition';
        if (/step|process|procedure|how to/.test(queryLower))
            return 'instruction';
        return 'general';
    }
}
let ragAgentServiceInstance = null;
export const getRAGAgentService = () => {
    if (!ragAgentServiceInstance) {
        ragAgentServiceInstance = new RAGAgentService();
    }
    return ragAgentServiceInstance;
};
//# sourceMappingURL=rag-agent.js.map