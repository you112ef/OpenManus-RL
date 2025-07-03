const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const path = require('path');
require('dotenv').config();

const AIModel = require('./src/models/AIModel');
const ChatService = require('./src/services/ChatService');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(helmet({
  contentSecurityPolicy: false,
  crossOriginEmbedderPolicy: false
}));
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Initialize AI Model and Chat Service
const aiModel = new AIModel();
const chatService = new ChatService(aiModel);

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'online', 
    model: aiModel.getModelStatus(),
    timestamp: new Date().toISOString()
  });
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, conversationId } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const response = await chatService.processMessage(message, conversationId);
    res.json(response);
  } catch (error) {
    console.error('Chat API Error:', error);
    res.status(500).json({ error: 'خطأ في معالجة الرسالة' });
  }
});

// Socket.io connection handling
io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('send-message', async (data) => {
    try {
      const { message, conversationId } = data;
      
      // Send typing indicator
      socket.emit('bot-typing', true);
      
      const response = await chatService.processMessage(message, conversationId);
      
      // Stop typing indicator
      socket.emit('bot-typing', false);
      
      // Send response
      socket.emit('receive-message', response);
    } catch (error) {
      console.error('Socket message error:', error);
      socket.emit('bot-typing', false);
      socket.emit('error', { message: 'خطأ في معالجة الرسالة' });
    }
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});

// Initialize AI Model
aiModel.initialize().then(() => {
  console.log('✅ AI Model initialized successfully');
}).catch((error) => {
  console.error('❌ AI Model initialization failed:', error);
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`🚀 YOUSEF SH AI Agent server running on port ${PORT}`);
  console.log(`🌐 Access the application at http://localhost:${PORT}`);
});