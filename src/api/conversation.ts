import { Router } from 'express';
import { getSessionManagementService } from '../services/session-manager.js';

const router = Router();
const sessionService = getSessionManagementService();

// Get full history for a session
router.get('/history/:sessionId', async (req, res) => {
    const { sessionId } = req.params;
    const messages = await sessionService.getSessionMessages(sessionId);
    res.json({
        sessionId,
        messages,
        totalMessages: messages.length,
    });
});

// Get recent context
router.get('/context/:sessionId', async (req, res) => {
    const { sessionId } = req.params;
    const context = await sessionService.getConversationContext(sessionId);
    const session = await sessionService.getSession(sessionId);

    res.json({
        sessionId,
        contextMessages: context,
        currentTopic: session?.activeTopic,
        conversationDepth: session?.conversationDepth || 0,
    });
});

// Reset conversation context
router.post('/reset-context/:sessionId', async (req, res) => {
    const { sessionId } = req.params;
    const success = await sessionService.clearSessionHistory(sessionId);
    if (!success) return res.status(500).json({ error: 'Failed to reset context' });

    res.json({ message: 'Conversation context reset successfully', sessionId });
});

export default router;
