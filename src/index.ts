import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import { pino } from 'pino';
import chatRouter from './api/chat.js';
import healthRouter from './api/health.js';
import textSelectionRouter from './api/text-selection.js';
import conversationRouter from './api/conversation.js';
import { WebSocketServer } from 'ws';
import { handleWebSocketConnection } from './services/websocket.js';

dotenv.config();

const logger = pino({
    transport: {
        target: 'pino-pretty',
    },
});

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/chat', chatRouter);
app.use('/api/health', healthRouter);
app.use('/api/text-selection', textSelectionRouter);
app.use('/api/conversation', conversationRouter);

// Global Error Handler
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    logger.error(err);
    res.status(500).json({ error: 'Internal Server Error' });
});

const server = app.listen(PORT, () => {
    logger.info(`Server is running on port ${PORT}`);
});

// WebSocket Server
const wss = new WebSocketServer({ server });
wss.on('connection', (ws, req) => {
    handleWebSocketConnection(ws, req);
});

export default app;
