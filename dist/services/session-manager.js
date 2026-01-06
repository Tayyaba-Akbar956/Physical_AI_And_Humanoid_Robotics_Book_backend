import prisma from '../db/prisma.js';
export class SessionManagementService {
    constructor() { }
    async createSession(studentId, currentModuleContext) {
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
        }
        catch (error) {
            console.error(`Error creating session for student ${studentId}:`, error);
            return null;
        }
    }
    async getSession(sessionId) {
        try {
            return await prisma.chatSession.findUnique({
                where: { id: sessionId },
            });
        }
        catch (error) {
            console.error(`Error retrieving session ${sessionId}:`, error);
            return null;
        }
    }
    async updateSession(sessionId, data) {
        try {
            return await prisma.chatSession.update({
                where: { id: sessionId },
                data: {
                    ...data,
                    updatedAt: new Date(),
                },
            });
        }
        catch (error) {
            console.error(`Error updating session ${sessionId}:`, error);
            return null;
        }
    }
    async endSession(sessionId) {
        return this.updateSession(sessionId, { isActive: false });
    }
    async getStudentSessions(studentId, activeOnly = false) {
        try {
            return await prisma.chatSession.findMany({
                where: {
                    studentId,
                    ...(activeOnly ? { isActive: true } : {}),
                },
                orderBy: { updatedAt: 'desc' },
            });
        }
        catch (error) {
            console.error(`Error retrieving sessions for student ${studentId}:`, error);
            return [];
        }
    }
    async addMessageToSession(params) {
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
        }
        catch (error) {
            console.error(`Error adding message to session ${params.sessionId}:`, error);
            return null;
        }
    }
    async getSessionMessages(sessionId, limit = 50) {
        try {
            return await prisma.message.findMany({
                where: { sessionId },
                orderBy: { timestamp: 'asc' },
                take: limit,
            });
        }
        catch (error) {
            console.error(`Error retrieving messages for session ${sessionId}:`, error);
            return [];
        }
    }
    async clearSessionHistory(sessionId) {
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
        }
        catch (error) {
            console.error(`Error clearing history for session ${sessionId}:`, error);
            return false;
        }
    }
    async getConversationContext(sessionId, numMessages = 10) {
        try {
            return await prisma.message.findMany({
                where: { sessionId },
                orderBy: { timestamp: 'desc' },
                take: numMessages,
            }).then((msgs) => msgs.reverse());
        }
        catch (error) {
            console.error(`Error retrieving context for session ${sessionId}:`, error);
            return [];
        }
    }
    async getModuleContextHistory(studentId, limit = 10) {
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
            return sessions.map((s) => ({
                sessionId: s.id,
                moduleContext: s.currentModuleContext,
                createdAt: s.createdAt,
                updatedAt: s.updatedAt,
                isActive: s.isActive,
                messageCount: s._count.messages,
                conversationDepth: s.conversationDepth,
                activeTopic: s.activeTopic,
            }));
        }
        catch (error) {
            console.error(`Error retrieving module history for student ${studentId}:`, error);
            return [];
        }
    }
}
let sessionServiceInstance = null;
export const getSessionManagementService = () => {
    if (!sessionServiceInstance) {
        sessionServiceInstance = new SessionManagementService();
    }
    return sessionServiceInstance;
};
//# sourceMappingURL=session-manager.js.map