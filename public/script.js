class YousefSHAgent {
  constructor() {
    this.socket = io();
    this.conversationId = this.generateUUID();
    this.messageCount = 0;
    this.isTyping = false;
    
    this.initializeElements();
    this.setupEventListeners();
    this.initializeSocket();
    this.checkStatus();
  }

  initializeElements() {
    // Core elements
    this.messageInput = document.getElementById('messageInput');
    this.sendButton = document.getElementById('sendButton');
    this.messagesContainer = document.getElementById('messages');
    this.welcomeSection = document.getElementById('welcomeSection');
    this.messagesSection = document.getElementById('messagesContainer');
    this.charCount = document.getElementById('charCount');
    this.typingIndicator = document.getElementById('typingIndicator');
    
    // Status elements
    this.statusIndicator = document.getElementById('statusIndicator');
    this.statusText = document.getElementById('statusText');
    this.statusDot = this.statusIndicator.querySelector('.status-dot');
    
    // Modal elements
    this.settingsModal = document.getElementById('settingsModal');
    this.settingsBtn = document.getElementById('settingsBtn');
    this.closeSettings = document.getElementById('closeSettings');
    this.newChatBtn = document.getElementById('newChatBtn');
    this.clearHistoryBtn = document.getElementById('clearHistory');
    
    // Status modal elements
    this.connectionStatus = document.getElementById('connectionStatus');
    this.aiMode = document.getElementById('aiMode');
    this.messageCountSpan = document.getElementById('messageCount');
    
    // Suggestion buttons
    this.suggestionBtns = document.querySelectorAll('.suggestion-btn');
  }

  setupEventListeners() {
    // Input events
    this.messageInput.addEventListener('input', () => this.handleInput());
    this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
    this.sendButton.addEventListener('click', () => this.sendMessage());
    
    // Control buttons
    this.settingsBtn.addEventListener('click', () => this.openSettings());
    this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
    this.newChatBtn.addEventListener('click', () => this.startNewChat());
    this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    
    // Suggestion buttons
    this.suggestionBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const message = btn.getAttribute('data-message');
        this.sendPredefinedMessage(message);
      });
    });
    
    // Modal click outside to close
    this.settingsModal.addEventListener('click', (e) => {
      if (e.target === this.settingsModal) {
        this.closeSettingsModal();
      }
    });
    
    // Auto-resize textarea
    this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
  }

  initializeSocket() {
    this.socket.on('connect', () => {
      console.log('Connected to server');
      this.updateStatus(true);
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from server');
      this.updateStatus(false);
    });

    this.socket.on('receive-message', (response) => {
      this.handleBotResponse(response);
    });

    this.socket.on('bot-typing', (isTyping) => {
      this.showTypingIndicator(isTyping);
    });

    this.socket.on('error', (error) => {
      console.error('Socket error:', error);
      this.showError(error.message || 'حدث خطأ في الاتصال');
    });
  }

  handleInput() {
    const value = this.messageInput.value;
    const length = value.length;
    
    // Update character count
    this.charCount.textContent = `${length}/4000`;
    
    // Enable/disable send button
    this.sendButton.disabled = length === 0 || length > 4000;
    
    // Update send button color based on state
    if (length > 0 && length <= 4000) {
      this.sendButton.style.opacity = '1';
    } else {
      this.sendButton.style.opacity = '0.5';
    }
  }

  handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!this.sendButton.disabled && !this.isTyping) {
        this.sendMessage();
      }
    }
  }

  autoResizeTextarea() {
    this.messageInput.style.height = 'auto';
    this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message || this.isTyping) return;

    // Show the chat interface if it's the first message
    if (this.messageCount === 0) {
      this.showChatInterface();
    }

    // Add user message to UI
    this.addMessage('user', message);
    
    // Clear input
    this.messageInput.value = '';
    this.handleInput();
    this.autoResizeTextarea();
    
    // Set typing state
    this.isTyping = true;
    this.sendButton.disabled = true;
    
    // Send message via socket
    this.socket.emit('send-message', {
      message: message,
      conversationId: this.conversationId
    });
    
    this.messageCount++;
    this.updateMessageCount();
  }

  sendPredefinedMessage(message) {
    this.messageInput.value = message;
    this.handleInput();
    this.sendMessage();
  }

  handleBotResponse(response) {
    this.isTyping = false;
    this.sendButton.disabled = this.messageInput.value.trim().length === 0;
    
    if (response.error) {
      this.showError(response.message);
    } else {
      this.addMessage('assistant', response.message);
    }
    
    // Update AI mode based on response metadata
    if (response.metadata && response.metadata.modelStatus) {
      this.updateAIMode(response.metadata.modelStatus);
    }
  }

  addMessage(sender, content) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${sender}`;
    
    const avatarEl = document.createElement('div');
    avatarEl.className = 'message-avatar';
    
    if (sender === 'user') {
      avatarEl.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="2"/>
        </svg>
      `;
    } else {
      avatarEl.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="currentColor"/>
          <path d="M19 15L19.75 18.5L23 19L19.75 19.5L19 23L18.25 19.5L15 19L18.25 18.5L19 15Z" fill="currentColor"/>
        </svg>
      `;
    }
    
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    
    const textEl = document.createElement('div');
    textEl.className = 'message-text';
    textEl.textContent = content;
    
    const timeEl = document.createElement('div');
    timeEl.className = 'message-time';
    timeEl.textContent = new Date().toLocaleTimeString('ar-EG', {
      hour: '2-digit',
      minute: '2-digit'
    });
    
    contentEl.appendChild(textEl);
    contentEl.appendChild(timeEl);
    messageEl.appendChild(avatarEl);
    messageEl.appendChild(contentEl);
    
    this.messagesContainer.appendChild(messageEl);
    this.scrollToBottom();
  }

  showTypingIndicator(show) {
    if (show) {
      this.typingIndicator.style.display = 'flex';
    } else {
      this.typingIndicator.style.display = 'none';
    }
  }

  showError(message) {
    this.addMessage('assistant', `خطأ: ${message}`);
  }

  showChatInterface() {
    this.welcomeSection.style.display = 'none';
    this.messagesSection.style.display = 'block';
  }

  scrollToBottom() {
    setTimeout(() => {
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }, 100);
  }

  updateStatus(isOnline) {
    if (isOnline) {
      this.statusDot.classList.remove('offline');
      this.statusDot.classList.add('online');
      this.statusText.textContent = 'متصل';
    } else {
      this.statusDot.classList.remove('online');
      this.statusDot.classList.add('offline');
      this.statusText.textContent = 'غير متصل';
    }
  }

  updateAIMode(modelStatus) {
    if (modelStatus.isOnline) {
      this.aiMode.textContent = 'متصل بالإنترنت';
      this.connectionStatus.textContent = 'متصل';
    } else {
      this.aiMode.textContent = 'وضع محلي';
      this.connectionStatus.textContent = 'غير متصل';
    }
  }

  updateMessageCount() {
    this.messageCountSpan.textContent = this.messageCount;
  }

  openSettings() {
    this.settingsModal.classList.add('show');
    this.checkStatus(); // Refresh status when opening settings
  }

  closeSettingsModal() {
    this.settingsModal.classList.remove('show');
  }

  startNewChat() {
    this.conversationId = this.generateUUID();
    this.messageCount = 0;
    this.messagesContainer.innerHTML = '';
    this.welcomeSection.style.display = 'block';
    this.messagesSection.style.display = 'none';
    this.updateMessageCount();
    this.messageInput.value = '';
    this.handleInput();
  }

  clearHistory() {
    if (confirm('هل أنت متأكد من مسح تاريخ المحادثة؟')) {
      this.startNewChat();
      this.closeSettingsModal();
    }
  }

  async checkStatus() {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      
      this.updateStatus(data.status === 'online');
      this.updateAIMode(data.model);
      
    } catch (error) {
      console.error('Status check failed:', error);
      this.updateStatus(false);
    }
  }

  generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // Initialize typing effect for welcome message
  initializeTypingEffect() {
    const text = "مرحباً بك في YOUSEF SH";
    const element = document.querySelector('.welcome-section h2');
    let index = 0;
    
    element.textContent = '';
    
    function typeChar() {
      if (index < text.length) {
        element.textContent += text.charAt(index);
        index++;
        setTimeout(typeChar, 100);
      }
    }
    
    setTimeout(typeChar, 500);
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  const app = new YousefSHAgent();
  
  // Add some loading animation
  const loadingElements = document.querySelectorAll('.feature-card');
  loadingElements.forEach((element, index) => {
    element.style.opacity = '0';
    element.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
      element.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
      element.style.opacity = '1';
      element.style.transform = 'translateY(0)';
    }, 200 + (index * 100));
  });
  
  // Add welcome message typing effect
  setTimeout(() => {
    app.initializeTypingEffect();
  }, 1000);
  
  // Periodic status check
  setInterval(() => {
    app.checkStatus();
  }, 30000); // Check every 30 seconds
  
  console.log('🚀 YOUSEF SH AI Agent initialized successfully');
});

// Add service worker for offline functionality
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}