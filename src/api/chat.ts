import { Router } from 'express';
import { getRAGAgentService } from '../services/rag-agent.js';
import { getSessionManagementService } from '../services/session-manager.js';

const router = Router();
const ragAgent = getRAGAgentService();
const sessionService = getSessionManagementService();

// Create a new session
router.post('/session/create', async (req, res) => {
    const { student_id, module_context } = req.body;
    const studentId = student_id || req.body.studentId;
    const currentModuleContext = module_context || req.body.currentModuleContext;

    if (!studentId) return res.status(400).json({ error: 'student_id is required' });

    const session = await sessionService.createSession(studentId, currentModuleContext);
    if (!session) return res.status(500).json({ error: 'Failed to create session' });

    res.json({
        ...session,
        session_id: session.id, // For compatibility
    });
});

// Query the RAG agent
router.post('/query', async (req, res) => {
    const { message, query, session_id, sessionId, module_context, moduleContext, selected_text, selectedText } = req.body;
    const actualQuery = message || query;
    let actualSessionId = session_id || sessionId;
    const actualModuleContext = module_context || moduleContext;
    const actualSelectedText = selected_text || selectedText;

    if (!actualQuery) {
        return res.status(400).json({ error: 'message/query is required' });
    }

    // Auto-create session if not provided
    if (!actualSessionId) {
        const newSession = await sessionService.createSession('anonymous', actualModuleContext);
        if (!newSession) {
            return res.status(500).json({ error: 'Failed to create session' });
        }
        actualSessionId = newSession.id;
    }

    const conversationContext = await sessionService.getConversationContext(actualSessionId);
    const relevantContent = await ragAgent.getRelevantContent({ query: actualQuery, moduleFilter: actualModuleContext });

    const response = await ragAgent.generateResponse({
        query: actualQuery,
        context: relevantContent,
        selectedText: actualSelectedText,
        moduleContext: actualModuleContext,
        conversationContext,
    });

    // Save student message
    await sessionService.addMessageToSession({
        sessionId: actualSessionId,
        senderType: 'student',
        content: actualQuery,
    });

    // Save AI message
    await sessionService.addMessageToSession({
        sessionId: actualSessionId,
        senderType: 'ai_agent',
        content: response.response,
        citations: response.citations,
        topicAnchored: response.responseIntent,
    });

    res.json({
        ...response,
        message: response.response, // For embed-script.js
        response: response.response, // For chat-widget.js
        session_id: actualSessionId,
        timestamp: new Date().toISOString(),
    });
});

// Get session messages
router.get('/session/:sessionId', async (req, res) => {
    const { sessionId } = req.params;
    const messages = await sessionService.getSessionMessages(sessionId);
    res.json(messages);
});

// Clear session history
router.post('/session/:sessionId/clear', async (req, res) => {
    const { sessionId } = req.params;
    const success = await sessionService.clearSessionHistory(sessionId);
    res.json({ success });
});

export default router;
