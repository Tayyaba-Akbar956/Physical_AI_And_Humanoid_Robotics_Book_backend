import { Router } from 'express';
import { getRAGAgentService } from '../services/rag-agent.js';
import { getSessionManagementService } from '../services/session-manager.js';

const router = Router();
const ragAgent = getRAGAgentService();
const sessionService = getSessionManagementService();

// Detect and validate text selection (Mock for now, can be expanded)
router.post('/detect', async (req, res) => {
    const { selectedText, selected_text } = req.body;
    const text = selectedText || selected_text;

    if (!text || text.length < 20) {
        return res.json({
            validation_result: 'invalid',
            character_count: text?.length || 0,
            can_ask_query: false,
            suggestions: ['Selected text must be at least 20 characters']
        });
    }

    res.json({
        validation_result: 'valid',
        character_count: text.length,
        can_ask_query: true
    });
});

// Query about selected text
router.post('/query', async (req, res) => {
    const { sessionId, session_id, selectedText, selected_text, question, message, moduleId, module_id, chapterId, chapter_id } = req.body;
    const actualSessionId = sessionId || session_id;
    const actualSelectedText = selectedText || selected_text;
    const actualQuestion = question || message;
    const actualModuleId = moduleId || module_id;

    if (!actualSessionId || !actualSelectedText || !actualQuestion) {
        return res.status(400).json({ error: 'sessionId, selectedText, and question/message are required' });
    }

    const relevantContent = await ragAgent.getRelevantContent({
        query: actualQuestion,
        moduleFilter: actualModuleId,
    });

    const response = await ragAgent.generateResponse({
        query: actualQuestion,
        context: relevantContent,
        selectedText: actualSelectedText,
        moduleContext: actualModuleId,
    });

    // Save student message with selected text ref
    await sessionService.addMessageToSession({
        sessionId: actualSessionId,
        senderType: 'student',
        content: `Regarding selected text: '${actualSelectedText.substring(0, 100)}...', I ask: ${actualQuestion}`,
    });

    // Save AI message
    await sessionService.addMessageToSession({
        sessionId: actualSessionId,
        senderType: 'ai_agent',
        content: response.response,
        citations: response.citations,
    });

    res.json({
        sessionId: actualSessionId,
        session_id: actualSessionId,
        message: response.response,
        response: response.response,
        citations: response.citations,
        timestamp: new Date().toISOString(),
    });
});

export default router;
