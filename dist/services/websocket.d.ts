import { WebSocket } from 'ws';
import { IncomingMessage } from 'http';
export declare const handleWebSocketConnection: (ws: WebSocket, req: IncomingMessage) => void;
