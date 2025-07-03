const { v4: uuidv4 } = require('uuid');

class ChatService {
  constructor(aiModel) {
    this.aiModel = aiModel;
    this.activeSessions = new Map();
    this.messageQueue = new Map();
  }

  async processMessage(message, conversationId = null) {
    try {
      // Generate conversation ID if not provided
      if (!conversationId) {
        conversationId = uuidv4();
      }

      // Validate and sanitize message
      const sanitizedMessage = this.sanitizeMessage(message);
      if (!sanitizedMessage) {
        throw new Error('Invalid message content');
      }

      // Create or get session
      if (!this.activeSessions.has(conversationId)) {
        this.activeSessions.set(conversationId, {
          id: conversationId,
          startTime: new Date(),
          messageCount: 0,
          lastActivity: new Date()
        });
      }

      const session = this.activeSessions.get(conversationId);
      session.messageCount++;
      session.lastActivity = new Date();

      // Generate AI response
      const aiResponse = await this.aiModel.generateResponse(sanitizedMessage, conversationId);

      // Format response
      const response = {
        id: uuidv4(),
        conversationId: conversationId,
        message: aiResponse,
        timestamp: new Date().toISOString(),
        sender: 'assistant',
        metadata: {
          modelStatus: this.aiModel.getModelStatus(),
          responseTime: Date.now(),
          messageCount: session.messageCount
        }
      };

      return response;
    } catch (error) {
      console.error('Chat service error:', error);
      return this.createErrorResponse(conversationId, error.message);
    }
  }

  sanitizeMessage(message) {
    if (typeof message !== 'string') {
      return null;
    }

    // Remove potentially harmful content
    const sanitized = message
      .trim()
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
      .replace(/<[^>]*>/g, '')
      .substring(0, 4000); // Limit message length

    return sanitized.length > 0 ? sanitized : null;
  }

  createErrorResponse(conversationId, errorMessage) {
    return {
      id: uuidv4(),
      conversationId: conversationId || 'error',
      message: 'أعتذر، حدث خطأ في معالجة رسالتك. يرجى المحاولة مرة أخرى.',
      timestamp: new Date().toISOString(),
      sender: 'assistant',
      error: true,
      metadata: {
        errorMessage: errorMessage,
        modelStatus: this.aiModel.getModelStatus()
      }
    };
  }

  getSessionInfo(conversationId) {
    return this.activeSessions.get(conversationId) || null;
  }

  getAllSessions() {
    return Array.from(this.activeSessions.values());
  }

  clearSession(conversationId) {
    this.activeSessions.delete(conversationId);
    this.aiModel.clearConversation(conversationId);
  }

  clearAllSessions() {
    this.activeSessions.clear();
    this.aiModel.clearAllConversations();
  }

  // Clean up inactive sessions (older than 1 hour)
  cleanupInactiveSessions() {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    
    for (const [sessionId, session] of this.activeSessions.entries()) {
      if (session.lastActivity < oneHourAgo) {
        this.clearSession(sessionId);
      }
    }
  }

  getStats() {
    const sessions = Array.from(this.activeSessions.values());
    return {
      totalSessions: sessions.length,
      totalMessages: sessions.reduce((sum, session) => sum + session.messageCount, 0),
      modelStatus: this.aiModel.getModelStatus(),
      activeSessions: sessions.filter(session => {
        const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
        return session.lastActivity > fiveMinutesAgo;
      }).length
    };
  }
}

// Add UUID generation if not available
if (!ChatService.prototype.uuidv4) {
  ChatService.prototype.uuidv4 = function() {
    try {
      return require('uuid').v4();
    } catch (error) {
      // Fallback UUID generation
      return 'xxxx-xxxx-4xxx-yxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    }
  };
}

module.exports = ChatService;