import { WebSocket } from 'ws';
import { IncomingMessage } from 'http';
import { getRAGAgentService } from './rag-agent.js';
import { getSessionManagementService } from './session-manager.js';

const ragAgent = getRAGAgentService();
const sessionService = getSessionManagementService();

export const handleWebSocketConnection = (ws: WebSocket, req: IncomingMessage) => {
    const url = new URL(req.url || '', `http://${req.headers.host}`);
    const sessionId = url.pathname.split('/').pop();

    if (!sessionId) {
        ws.close(1003, 'Session ID required');
        return;
    }

    ws.on('message', async (data) => {
        try {
            const message = JSON.parse(data.toString());
            const { type, query, moduleContext, selectedText } = message;

            if (type === 'query') {
                const relevantContent = await ragAgent.getRelevantContent({ query, moduleFilter: moduleContext });
                const response = await ragAgent.generateResponse({
                    query,
                    context: relevantContent,
                    selectedText,
                    moduleContext,
                });

                ws.send(JSON.stringify({
                    type: 'response',
                    message: response.response,
                    citations: response.citations,
                }));

                // Persist messages in background
                sessionService.addMessageToSession({
                    sessionId,
                    senderType: 'student',
                    content: query,
                });
                sessionService.addMessageToSession({
                    sessionId,
                    senderType: 'ai_agent',
                    content: response.response,
                    citations: response.citations,
                });
            }
        } catch (error) {
            ws.send(JSON.stringify({ type: 'error', message: 'Failed to process message' }));
        }
    });

    ws.on('close', () => {
        console.log(`WebSocket connection closed for session ${sessionId}`);
    });
};
