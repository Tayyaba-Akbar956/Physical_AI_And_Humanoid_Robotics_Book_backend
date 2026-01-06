import express from 'express';
import * as cors from 'cors';
import * as helmet from 'helmet';
import dotenv from 'dotenv';
import { pino } from 'pino';
import chatRouter from './api/chat.js';
import healthRouter from './api/health.js';
import textSelectionRouter from './api/text-selection.js';
import conversationRouter from './api/conversation.js';
import { WebSocketServer } from 'ws';
import { handleWebSocketConnection } from './services/websocket.js';

dotenv.config();

const isDev = process.env.NODE_ENV !== 'production' && !process.env.VERCEL;

const logger = pino(isDev ? {
    transport: {
        target: 'pino-pretty',
    },
} : {});

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use((helmet as any).default ? (helmet as any).default() : (helmet as any)());
app.use((cors as any).default ? (cors as any).default() : (cors as any)());
app.use(express.json());

// Routes
app.get('/', (req, res) => {
    res.json({ status: 'active', message: 'Physical AI Book Backend is running' });
});
app.use('/api/chat', chatRouter);
app.use('/api/health', healthRouter);
app.use('/api/text-selection', textSelectionRouter);
app.use('/api/conversation', conversationRouter);

// Global Error Handler
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    logger.error(err);
    res.status(500).json({ error: 'Internal Server Error' });
});

let server: any;
if (!process.env.VERCEL) {
    server = app.listen(PORT, () => {
        logger.info(`Server is running on port ${PORT}`);
    });
}

// WebSocket Server (Note: WebSockets will not work on Vercel Serverless Functions)
if (server && !process.env.VERCEL) {
    const wss = new WebSocketServer({ server });
    wss.on('connection', (ws, req) => {
        handleWebSocketConnection(ws, req);
    });
}

export default app;
