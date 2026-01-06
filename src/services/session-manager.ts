import { PrismaClient } from '@prisma/client';
import prisma from '../db/prisma.js';
import { v4 as uuidv4 } from 'uuid';

export class SessionManagementService {
    constructor() { }

    async createSession(studentId: string, currentModuleContext?: string) {
        try {
            // Verify student exists
            const student = await prisma.student.findUnique({
                where: { id: studentId },
            });

            if (!student) {
                console.warn(`Student with ID ${studentId} not found`);
                return null;
            }

            // Create new session
            const session = await prisma.chatSession.create({
                data: {
                    studentId,
                    currentModuleContext,
                    sessionMetadata: {},
                },
            });

            console.info(`Created new session ${session.id} for student ${studentId}, module ${currentModuleContext}`);
            return session;
        } catch (error) {
            console.error(`Error creating session for student ${studentId}:`, error);
            return null;
        }
    }

    async getSession(sessionId: string) {
        try {
            return await prisma.chatSession.findUnique({
                where: { id: sessionId },
            });
        } catch (error) {
            console.error(`Error retrieving session ${sessionId}:`, error);
            return null;
        }
    }

    async updateSession(sessionId: string, data: any) {
        try {
            return await prisma.chatSession.update({
                where: { id: sessionId },
                data: {
                    ...data,
                    updatedAt: new Date(),
                },
            });
        } catch (error) {
            console.error(`Error updating session ${sessionId}:`, error);
            return null;
        }
    }

    async endSession(sessionId: string) {
        return this.updateSession(sessionId, { isActive: false });
    }

    async getStudentSessions(studentId: string, activeOnly: boolean = false) {
        try {
            return await prisma.chatSession.findMany({
                where: {
                    studentId,
                    ...(activeOnly ? { isActive: true } : {}),
                },
                orderBy: { updatedAt: 'desc' },
            });
        } catch (error) {
            console.error(`Error retrieving sessions for student ${studentId}:`, error);
            return [];
        }
    }

    async addMessageToSession(params: {
        sessionId: string;
        senderType: 'student' | 'ai_agent';
        content: string;
        citations?: any;
        selectedTextRef?: string;
        parentMessageId?: string;
        topicAnchored?: string;
        followUpTo?: string;
    }) {
        try {
            const { sessionId, senderType, content, citations, selectedTextRef, parentMessageId, topicAnchored, followUpTo } = params;

            // Verify session exists and is active
            const session = await prisma.chatSession.findFirst({
                where: { id: sessionId, isActive: true },
            });

            if (!session) {
                console.warn(`Session with ID ${sessionId} not found or not active`);
                return null;
            }

            // Calculate the conversation turn
            const conversationTurn = await prisma.message.count({
                where: { sessionId },
            });

            // Create message
            const message = await prisma.message.create({
                data: {
                    sessionId,
                    senderType,
                    content,
                    citations,
                    selectedTextRef,
                    conversationTurn,
                    parentMessageId,
                    topicAnchored,
                    followUpTo,
                },
            });

            // Update the session's conversation context (simplified for now)
            if (topicAnchored) {
                await prisma.chatSession.update({
                    where: { id: sessionId },
                    data: {
                        activeTopic: topicAnchored,
                        lastInteractionAt: new Date(),
                        conversationDepth: Math.floor((conversationTurn + 1) / 2),
                    },
                });
            }

            return message;
        } catch (error) {
            console.error(`Error adding message to session ${params.sessionId}:`, error);
            return null;
        }
    }

    async getSessionMessages(sessionId: string, limit: number = 50) {
        try {
            return await prisma.message.findMany({
                where: { sessionId },
                orderBy: { timestamp: 'asc' },
                take: limit,
            });
        } catch (error) {
            console.error(`Error retrieving messages for session ${sessionId}:`, error);
            return [];
        }
    }

    async clearSessionHistory(sessionId: string) {
        try {
            await prisma.message.deleteMany({
                where: { sessionId },
            });

            await prisma.chatSession.update({
                where: { id: sessionId },
                data: {
                    updatedAt: new Date(),
                    conversationDepth: 0,
                    activeTopic: null,
                },
            });
            return true;
        } catch (error) {
            console.error(`Error clearing history for session ${sessionId}:`, error);
            return false;
        }
    }

    async getConversationContext(sessionId: string, numMessages: number = 10) {
        try {
            return await prisma.message.findMany({
                where: { sessionId },
                orderBy: { timestamp: 'desc' },
                take: numMessages,
            }).then((msgs: any[]) => msgs.reverse());
        } catch (error) {
            console.error(`Error retrieving context for session ${sessionId}:`, error);
            return [];
        }
    }

    async getModuleContextHistory(studentId: string, limit: number = 10) {
        try {
            const sessions = await prisma.chatSession.findMany({
                where: { studentId },
                orderBy: { updatedAt: 'desc' },
                take: limit,
                include: {
                    _count: {
                        select: { messages: true },
                    },
                },
            });

            return sessions.map((s: any) => ({
                sessionId: s.id,
                moduleContext: s.currentModuleContext,
                createdAt: s.createdAt,
                updatedAt: s.updatedAt,
                isActive: s.isActive,
                messageCount: s._count.messages,
                conversationDepth: s.conversationDepth,
                activeTopic: s.activeTopic,
            }));
        } catch (error) {
            console.error(`Error retrieving module history for student ${studentId}:`, error);
            return [];
        }
    }
}

let sessionServiceInstance: SessionManagementService | null = null;

export const getSessionManagementService = (): SessionManagementService => {
    if (!sessionServiceInstance) {
        sessionServiceInstance = new SessionManagementService();
    }
    return sessionServiceInstance;
};
