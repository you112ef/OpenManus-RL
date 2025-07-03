const axios = require('axios');

class AIModel {
  constructor() {
    this.isOnline = true;
    this.localModel = null;
    this.modelStatus = 'initializing';
    this.conversationHistory = new Map();
    
    // Configuration
    this.config = {
      openai: {
        apiKey: process.env.OPENAI_API_KEY,
        model: 'gpt-3.5-turbo',
        baseURL: 'https://api.openai.com/v1'
      },
      local: {
        enabled: true,
        modelPath: './models/local-model',
        maxTokens: 2048
      }
    };

    // Arabic prompts and responses
    this.systemPrompts = {
      arabic: `أنت مساعد ذكي اسمه "يوسف شاهين" (YOUSEF SH). أنت مفيد ومهذب وتجيب باللغة العربية بشكل واضح ومفهوم. 
      تساعد المستخدمين في مختلف المواضيع والاستفسارات. كن ودودًا ومساعدًا دائمًا.`,
      english: `You are an intelligent assistant named "YOUSEF SH". You are helpful, polite, and respond clearly. 
      You assist users with various topics and inquiries. Always be friendly and helpful.`
    };

    this.offlineResponses = [
      "أعتذر، أعمل حاليًا في الوضع المحلي. كيف يمكنني مساعدتك؟",
      "مرحبًا! أعمل حاليًا بدون اتصال بالإنترنت، لكن يمكنني مساعدتك في الأسئلة العامة.",
      "أهلاً وسهلاً! أعمل في الوضع المحلي حاليًا. ما الذي تحتاج إلى مساعدة فيه؟"
    ];
  }

  async initialize() {
    try {
      // Check internet connectivity
      await this.checkConnectivity();
      
      if (this.isOnline) {
        this.modelStatus = 'online';
        console.log('🌐 AI Model running in online mode');
      } else {
        await this.initializeLocalModel();
        this.modelStatus = 'offline';
        console.log('🔌 AI Model running in offline mode');
      }
    } catch (error) {
      console.error('Model initialization error:', error);
      this.modelStatus = 'error';
      this.isOnline = false;
    }
  }

  async checkConnectivity() {
    try {
      const response = await axios.get('https://api.openai.com/v1/models', {
        headers: {
          'Authorization': `Bearer ${this.config.openai.apiKey || 'test'}`
        },
        timeout: 5000
      });
      this.isOnline = true;
    } catch (error) {
      console.log('No internet connection or API unavailable, switching to offline mode');
      this.isOnline = false;
    }
  }

  async initializeLocalModel() {
    try {
      // Simulate local model initialization
      // In a real implementation, you would load a local model here
      // using libraries like transformers.js, tensorflow.js, or onnx
      
      this.localModel = {
        initialized: true,
        name: 'Local Arabic Model',
        version: '1.0.0'
      };
      
      console.log('📦 Local model initialized successfully');
    } catch (error) {
      console.error('Local model initialization failed:', error);
      throw error;
    }
  }

  async generateResponse(message, conversationId = 'default') {
    try {
      // Get or create conversation history
      if (!this.conversationHistory.has(conversationId)) {
        this.conversationHistory.set(conversationId, []);
      }
      
      const history = this.conversationHistory.get(conversationId);
      
      let response;
      
      if (this.isOnline && this.config.openai.apiKey) {
        response = await this.generateOnlineResponse(message, history);
      } else {
        response = await this.generateOfflineResponse(message, history);
      }

      // Update conversation history
      history.push(
        { role: 'user', content: message },
        { role: 'assistant', content: response }
      );

      // Keep only last 10 exchanges to manage memory
      if (history.length > 20) {
        history.splice(0, history.length - 20);
      }

      return response;
    } catch (error) {
      console.error('Response generation error:', error);
      return this.getErrorResponse();
    }
  }

  async generateOnlineResponse(message, history) {
    try {
      const messages = [
        { role: 'system', content: this.systemPrompts.arabic },
        ...history,
        { role: 'user', content: message }
      ];

      const response = await axios.post(
        `${this.config.openai.baseURL}/chat/completions`,
        {
          model: this.config.openai.model,
          messages: messages,
          max_tokens: 1000,
          temperature: 0.7,
          top_p: 0.9
        },
        {
          headers: {
            'Authorization': `Bearer ${this.config.openai.apiKey}`,
            'Content-Type': 'application/json'
          },
          timeout: 30000
        }
      );

      return response.data.choices[0].message.content;
    } catch (error) {
      console.error('Online response generation failed:', error);
      // Fallback to offline mode
      this.isOnline = false;
      return this.generateOfflineResponse(message, history);
    }
  }

  async generateOfflineResponse(message, history) {
    // Simple rule-based responses for offline mode
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('مرحبا') || lowerMessage.includes('أهلا') || lowerMessage.includes('السلام')) {
      return "مرحبًا بك! أنا يوسف شاهين، مساعدك الذكي. كيف يمكنني مساعدتك اليوم؟";
    }
    
    if (lowerMessage.includes('كيف حالك') || lowerMessage.includes('أخبارك')) {
      return "أشكرك على السؤال! أعمل بشكل جيد وجاهز لمساعدتك. ما الذي تحتاج إليه؟";
    }
    
    if (lowerMessage.includes('من أنت') || lowerMessage.includes('اسمك')) {
      return "أنا يوسف شاهين (YOUSEF SH)، مساعد ذكي مطور للمساعدة في مختلف المهام والاستفسارات. أعمل حاليًا في الوضع المحلي.";
    }
    
    if (lowerMessage.includes('مساعدة') || lowerMessage.includes('help')) {
      return "بالطبع! يمكنني مساعدتك في:\n• الإجابة على الأسئلة العامة\n• تقديم المعلومات\n• المحادثة والدردشة\n• حل المشاكل البسيطة\n\nما الذي تحتاج إلى مساعدة فيه؟";
    }
    
    if (lowerMessage.includes('وقت') || lowerMessage.includes('تاريخ')) {
      const now = new Date();
      const arabicDate = now.toLocaleDateString('ar-EG', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
      const arabicTime = now.toLocaleTimeString('ar-EG');
      return `التاريخ الحالي: ${arabicDate}\nالوقت الحالي: ${arabicTime}`;
    }
    
    // Default response with some variation
    const randomResponse = this.offlineResponses[
      Math.floor(Math.random() * this.offlineResponses.length)
    ];
    
    return `${randomResponse}\n\nسؤالك: "${message}"\n\nعذرًا، أعمل حاليًا في الوضع المحلي وقدراتي محدودة. هل يمكنك إعادة صياغة السؤال أو طرح سؤال آخر؟`;
  }

  getErrorResponse() {
    return "أعتذر، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى لاحقًا.";
  }

  getModelStatus() {
    return {
      status: this.modelStatus,
      isOnline: this.isOnline,
      hasLocalModel: !!this.localModel,
      activeConversations: this.conversationHistory.size
    };
  }

  clearConversation(conversationId) {
    this.conversationHistory.delete(conversationId);
  }

  clearAllConversations() {
    this.conversationHistory.clear();
  }
}

module.exports = AIModel;