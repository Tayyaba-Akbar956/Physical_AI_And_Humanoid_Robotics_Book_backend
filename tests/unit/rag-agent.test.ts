import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RAGAgentService } from '../../src/services/rag-agent.js';
import { getGeminiService } from '../../src/services/gemini.js';
import { getSemanticSearchService } from '../../src/services/semantic-search.js';
import { getSessionManagementService } from '../../src/services/session-manager.js';

vi.mock('../../src/services/gemini.js');
vi.mock('../../src/services/semantic-search.js');
vi.mock('../../src/services/session-manager.js');

describe('RAGAgentService', () => {
    let ragAgent: RAGAgentService;
    let mockGemini: any;
    let mockSearch: any;
    let mockSession: any;

    beforeEach(() => {
        mockGemini = {
            generateResponse: vi.fn().mockResolvedValue('Mocked response'),
        };
        mockSearch = {
            searchWithModulePrioritization: vi.fn().mockResolvedValue([{ content: 'context', moduleId: '1', chapterId: '1' }]),
        };
        mockSession = {
            getConversationContext: vi.fn().mockResolvedValue([]),
            addMessageToSession: vi.fn(),
        };

        (getGeminiService as any).mockReturnValue(mockGemini);
        (getSemanticSearchService as any).mockReturnValue(mockSearch);
        (getSessionManagementService as any).mockReturnValue(mockSession);

        ragAgent = new RAGAgentService();
    });

    it('should generate a response based on context', async () => {
        const result = await ragAgent.generateResponse({
            query: 'What is ROS 2?',
            context: [{ content: 'ROS 2 is a middleware', moduleId: '2', chapterId: '1', sectionId: '1', contentType: 'text', hierarchyPath: 'm2/c1/s1' }],
        });

        expect(result.response).toBe('Mocked response');
        expect(result.citations).toHaveLength(1);
        expect(mockGemini.generateResponse).toHaveBeenCalled();
    });

    it('should retrieve relevant content', async () => {
        const content = await ragAgent.getRelevantContent({ query: 'ROS 2' });
        expect(content).toHaveLength(1);
        expect(mockSearch.searchWithModulePrioritization).toHaveBeenCalled();
    });
});
