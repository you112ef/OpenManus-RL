# YOUSEF SH - المساعد الذكي
### AI Agent with Local Model Integration

<div align="center">

![YOUSEF SH Logo](https://img.shields.io/badge/YOUSEF%20SH-AI%20Agent-6366f1?style=for-the-badge&logo=robot&logoColor=white)

[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat&logo=node.js&logoColor=white)](https://nodejs.org/)
[![Express](https://img.shields.io/badge/Express-4.18+-000000?style=flat&logo=express&logoColor=white)](https://expressjs.com/)
[![Socket.io](https://img.shields.io/badge/Socket.io-4.7+-010101?style=flat&logo=socket.io&logoColor=white)](https://socket.io/)
[![Arabic Support](https://img.shields.io/badge/Arabic-Supported-ff6b6b?style=flat&logo=google-translate&logoColor=white)]()

</div>

## 🌟 المميزات الرئيسية | Key Features

- 🤖 **مساعد ذكي متقدم** - Advanced AI Assistant
- 🌐 **يعمل بدون إنترنت** - Works Offline & Online
- 🔄 **تكامل النماذج المحلية** - Local AI Model Integration
- 🎨 **تصميم عصري** - Modern UI/UX Design
- 🌙 **الوضع المظلم** - Dark Theme
- 📱 **متجاوب مع الهواتف** - Mobile Responsive
- ⚡ **سريع ومتجاوب** - Fast & Responsive
- 🔒 **آمن ومحمي** - Secure & Private
- 🇸🇦 **دعم اللغة العربية** - Full Arabic Support

## 🚀 البدء السريع | Quick Start

### متطلبات النظام | System Requirements

- Node.js 18+ 
- npm أو yarn
- 4GB RAM (minimum)
- 1GB storage space

### التثبيت | Installation

1. **استنساخ المشروع | Clone the project:**
```bash
git clone <repository-url>
cd yousef-sh-ai-agent
```

2. **تثبيت المتطلبات | Install dependencies:**
```bash
npm install
```

3. **إعداد متغيرات البيئة | Setup environment variables:**
```bash
cp .env.example .env
```
أضف مفتاح OpenAI الخاص بك (اختياري للوضع المتصل):
```env
OPENAI_API_KEY=your_api_key_here
```

4. **تشغيل التطبيق | Run the application:**
```bash
# Development mode
npm run dev

# Production mode
npm start
```

5. **افتح التطبيق | Open the application:**
```
http://localhost:3000
```

## 🏗️ هيكل المشروع | Project Structure

```
yousef-sh-ai-agent/
├── public/                 # Frontend files
│   ├── index.html         # Main HTML file
│   ├── styles.css         # CSS styles
│   ├── script.js          # Frontend JavaScript
│   └── sw.js              # Service Worker
├── src/                   # Backend source code
│   ├── models/            # AI Models
│   │   └── AIModel.js     # Main AI model class
│   └── services/          # Business logic
│       └── ChatService.js # Chat service
├── server.js              # Main server file
├── package.json           # Dependencies
├── .env.example           # Environment variables template
└── README_YOUSEF_SH.md    # This file
```

## 🔧 إعدادات التطبيق | Configuration

### متغيرات البيئة | Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3000` |
| `OPENAI_API_KEY` | OpenAI API key (optional) | - |
| `LOCAL_MODEL_ENABLED` | Enable local model | `true` |
| `NODE_ENV` | Environment mode | `development` |

### أوضاع التشغيل | Operating Modes

#### 1. الوضع المتصل | Online Mode
- يستخدم OpenAI API
- يتطلب مفتاح API
- استجابات أكثر تقدماً

#### 2. الوضع المحلي | Offline Mode
- يعمل بدون إنترنت
- يستخدم نظام الاستجابات المحلي
- حماية كاملة للخصوصية

## 🎯 كيفية الاستخدام | How to Use

### 1. بدء محادثة جديدة | Starting a New Chat
- افتح التطبيق في المتصفح
- اكتب رسالتك في حقل النص
- اضغط Enter أو زر الإرسال

### 2. الأسئلة المقترحة | Suggested Questions
- انقر على الأسئلة المقترحة للبدء السريع
- مثل: "مرحباً، كيف حالك؟" أو "من أنت؟"

### 3. الإعدادات | Settings
- انقر على أيقونة الإعدادات في الأعلى
- راجع حالة الاتصال ووضع الذكاء الاصطناعي
- امسح تاريخ المحادثة حسب الحاجة

### 4. محادثة جديدة | New Chat
- انقر على أيقونة "+" لبدء محادثة جديدة
- سيتم مسح المحادثة الحالية

## 🔌 الميزات التقنية | Technical Features

### Frontend
- **HTML5** مع دعم RTL للعربية
- **CSS3** مع متغيرات وانيميشن
- **JavaScript ES6+** مع Socket.io
- **Service Worker** للعمل بدون إنترنت
- **Responsive Design** لجميع الأجهزة

### Backend
- **Node.js** مع Express.js
- **Socket.io** للاتصال المباشر
- **AI Model Integration** 
- **RESTful API** endpoints
- **Error Handling** شامل

### Security
- **Input Sanitization** تنظيف المدخلات
- **Rate Limiting** حد المعدل
- **CORS Protection** حماية CORS
- **Helmet.js** للأمان

## 🔍 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main application page |
| `GET` | `/api/health` | Health check & status |
| `POST` | `/api/chat` | Send chat message |

### Socket.io Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `send-message` | Client → Server | Send user message |
| `receive-message` | Server → Client | Receive AI response |
| `bot-typing` | Server → Client | Typing indicator |

## 🎨 التخصيص | Customization

### تغيير الألوان | Changing Colors
أضف في `styles.css`:
```css
:root {
  --primary-color: #your-color;
  --background-dark: #your-background;
}
```

### إضافة أسئلة مقترحة | Adding Suggested Questions
في `index.html` أضف:
```html
<button class="suggestion-btn" data-message="سؤالك الجديد">
  سؤالك الجديد
</button>
```

### تخصيص الردود المحلية | Customizing Local Responses
في `src/models/AIModel.js` أضف:
```javascript
if (lowerMessage.includes('كلمتك المفتاحية')) {
  return "ردك المخصص هنا";
}
```

## 🐛 استكشاف الأخطاء | Troubleshooting

### مشاكل شائعة | Common Issues

**1. التطبيق لا يعمل:**
```bash
# تحقق من إصدار Node.js
node --version  # يجب أن يكون 18+

# أعد تثبيت المتطلبات
rm -rf node_modules package-lock.json
npm install
```

**2. خطأ في الاتصال:**
- تأكد من أن المنفذ 3000 غير مستخدم
- تحقق من إعدادات الجدار الناري

**3. لا يعمل في الوضع المحلي:**
- تأكد من `LOCAL_MODEL_ENABLED=true` في `.env`
- أعد تشغيل الخادم

## 🔄 التحديثات | Updates

### v1.0.0
- ✅ واجهة مستخدم عصرية
- ✅ دعم الوضع المحلي والمتصل
- ✅ دعم اللغة العربية الكامل
- ✅ Service Worker للعمل بدون إنترنت
- ✅ تصميم متجاوب للهواتف

### خطط مستقبلية | Future Plans
- 🔄 تحسين النموذج المحلي
- 🔄 إضافة دعم الملفات
- 🔄 تطبيق هاتف محمول
- 🔄 دعم المزيد من اللغات

## 🤝 المساهمة | Contributing

نرحب بالمساهمات! يرجى:

1. عمل Fork للمشروع
2. إنشاء branch جديد (`git checkout -b feature/AmazingFeature`)
3. Commit التغييرات (`git commit -m 'Add some AmazingFeature'`)
4. Push إلى Branch (`git push origin feature/AmazingFeature`)
5. فتح Pull Request

## 📝 الترخيص | License

هذا المشروع مرخص تحت رخصة MIT. انظر ملف `LICENSE` للتفاصيل.

## 📞 الدعم | Support

- **Email:** support@yousefsh.com
- **GitHub Issues:** [أفتح issue جديد](../../issues)
- **Discord:** [انضم لمجتمعنا](#)

## 🙏 شكر وتقدير | Acknowledgments

- فريق OpenManus-RL للإلهام والكود الأساسي
- مجتمع العربي للذكاء الاصطناعي
- جميع المساهمين والمختبرين

---

<div align="center">

**طُور بـ ❤️ لخدمة المجتمع العربي**

Made with ❤️ for the Arabic Community

</div>