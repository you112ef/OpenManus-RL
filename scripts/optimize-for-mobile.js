const fs = require('fs');
const path = require('path');

console.log('🔧 Optimizing YOUSEF SH for Android...');

// Paths
const cordovaWwwPath = path.join(__dirname, '..', 'cordova', 'www');
const indexPath = path.join(cordovaWwwPath, 'index.html');
const stylesPath = path.join(cordovaWwwPath, 'styles.css');
const scriptPath = path.join(cordovaWwwPath, 'script.js');

// Ensure cordova/www directory exists
if (!fs.existsSync(cordovaWwwPath)) {
    fs.mkdirSync(cordovaWwwPath, { recursive: true });
    console.log('📁 Created cordova/www directory');
}

// Copy files from public to cordova/www
const publicPath = path.join(__dirname, '..', 'public');
if (fs.existsSync(publicPath)) {
    copyRecursiveSync(publicPath, cordovaWwwPath);
    console.log('📋 Copied files to cordova/www');
}

// Function to copy files recursively
function copyRecursiveSync(src, dest) {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();
    
    if (isDirectory) {
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest);
        }
        fs.readdirSync(src).forEach(function(childItemName) {
            copyRecursiveSync(path.join(src, childItemName), path.join(dest, childItemName));
        });
    } else {
        fs.copyFileSync(src, dest);
    }
}

// 1. Optimize HTML for mobile
if (fs.existsSync(indexPath)) {
    let htmlContent = fs.readFileSync(indexPath, 'utf8');
    
    // Add Cordova script
    htmlContent = htmlContent.replace(
        '<script src="/socket.io/socket.io.js"></script>',
        `<script type="text/javascript" src="cordova.js"></script>
        <script src="/socket.io/socket.io.js"></script>`
    );
    
    // Add mobile viewport meta tags
    htmlContent = htmlContent.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        `<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, shrink-to-fit=no">
        <meta name="format-detection" content="telephone=no">
        <meta name="msapplication-tap-highlight" content="no">
        <meta http-equiv="Content-Security-Policy" content="default-src 'self' data: https: 'unsafe-inline' 'unsafe-eval'; connect-src 'self' https: wss: ws:; media-src 'self' https: data:">`
    );
    
    // Remove service worker registration for mobile app
    htmlContent = htmlContent.replace(
        /\/\/ Add service worker.*?}\s*}\s*}/gs,
        '// Service Worker disabled in mobile app'
    );
    
    fs.writeFileSync(indexPath, htmlContent);
    console.log('📱 Optimized HTML for mobile');
}

// 2. Optimize CSS for mobile
if (fs.existsSync(stylesPath)) {
    let cssContent = fs.readFileSync(stylesPath, 'utf8');
    
    // Add mobile-specific styles
    const mobileCSS = `
/* Mobile App Specific Styles */
body {
    -webkit-touch-callout: none;
    -webkit-text-size-adjust: none;
    -webkit-user-select: none;
    -webkit-tap-highlight-color: rgba(0,0,0,0);
    overscroll-behavior: none;
}

/* Fix for mobile keyboard */
.input-area {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
}

/* Better touch targets */
button, .suggestion-btn, .control-btn {
    min-height: 44px;
    min-width: 44px;
}

/* Prevent zoom on input focus */
input, textarea, select {
    font-size: 16px !important;
}

/* Android specific fixes */
.android .header {
    padding-top: env(safe-area-inset-top, 0);
}

.android .input-area {
    padding-bottom: env(safe-area-inset-bottom, 0);
}

/* Hide scrollbars on mobile */
::-webkit-scrollbar {
    display: none;
}

/* Improve touch scrolling */
.messages-container, .main-content {
    -webkit-overflow-scrolling: touch;
    overflow-scrolling: touch;
}

/* Mobile-specific animations */
@media (max-width: 768px) {
    .message {
        animation: slideInMobile 0.2s ease-out;
    }
    
    @keyframes slideInMobile {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
}

/* Status bar styling */
.status-bar-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: env(safe-area-inset-top, 0);
    background: var(--background-dark);
    z-index: 1000;
}
    `;
    
    cssContent += mobileCSS;
    fs.writeFileSync(stylesPath, cssContent);
    console.log('🎨 Optimized CSS for mobile');
}

// 3. Optimize JavaScript for mobile
if (fs.existsSync(scriptPath)) {
    let jsContent = fs.readFileSync(scriptPath, 'utf8');
    
    // Add Cordova-specific JavaScript
    const mobileJS = `
// Cordova Device Ready Event
document.addEventListener('deviceready', function() {
    console.log('📱 Cordova device ready');
    
    // Set Android class for platform-specific styling
    if (device.platform === 'Android') {
        document.body.classList.add('android');
    }
    
    // Handle Android back button
    document.addEventListener('backbutton', function(e) {
        e.preventDefault();
        
        // If modal is open, close it
        const modal = document.querySelector('.modal.show');
        if (modal) {
            modal.classList.remove('show');
            return;
        }
        
        // If in chat, go back to welcome
        const messagesContainer = document.getElementById('messagesContainer');
        const welcomeSection = document.getElementById('welcomeSection');
        
        if (messagesContainer.style.display !== 'none') {
            // Ask user if they want to start new chat
            if (confirm('هل تريد بدء محادثة جديدة؟')) {
                if (window.app) {
                    window.app.startNewChat();
                }
            }
        } else {
            // Exit app
            if (confirm('هل تريد إغلاق التطبيق؟')) {
                navigator.app.exitApp();
            }
        }
    }, false);
    
    // Handle network status
    if (navigator.connection) {
        function updateConnectionStatus() {
            const networkState = navigator.connection.type;
            const isOnline = networkState !== Connection.NONE;
            
            if (window.app) {
                window.app.updateStatus(isOnline);
            }
        }
        
        document.addEventListener('online', updateConnectionStatus, false);
        document.addEventListener('offline', updateConnectionStatus, false);
        updateConnectionStatus();
    }
    
    // Initialize the app after device is ready
    if (window.YousefSHAgent) {
        window.app = new YousefSHAgent();
    }
}, false);

// Prevent context menu on long press
document.addEventListener('contextmenu', function(e) {
    e.preventDefault();
}, false);

// Prevent selection
document.addEventListener('selectstart', function(e) {
    e.preventDefault();
}, false);

// Handle status bar
if (StatusBar) {
    StatusBar.styleDefault();
    StatusBar.backgroundColorByHexString('#0f0f23');
}
    `;
    
    // Modify the existing initialization to wait for device ready
    jsContent = jsContent.replace(
        "document.addEventListener('DOMContentLoaded', () => {",
        `// Device ready handler for Cordova
function initializeApp() {`
    );
    
    jsContent = jsContent.replace(
        "});",
        `}

// Initialize based on environment
if (typeof cordova !== 'undefined') {
    // Mobile app - wait for device ready
    console.log('📱 Running as mobile app');
} else {
    // Web app - use DOM ready
    document.addEventListener('DOMContentLoaded', initializeApp);
}

${mobileJS}`
    );
    
    fs.writeFileSync(scriptPath, jsContent);
    console.log('⚡ Optimized JavaScript for mobile');
}

// 4. Create mobile-specific configuration files
const configFiles = {
    'cordova/res/android/xml/network_security_config.xml': `<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">localhost</domain>
        <domain includeSubdomains="true">127.0.0.1</domain>
        <domain includeSubdomains="true">10.0.2.2</domain>
    </domain-config>
</network-security-config>`,
    
    'cordova/res/android/values/strings.xml': `<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">YOUSEF SH</string>
    <string name="launcher_name">YOUSEF SH</string>
    <string name="activity_name">YOUSEF SH - المساعد الذكي</string>
</resources>`,
    
    'cordova/res/android/values/colors.xml': `<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="colorPrimary">#6366f1</color>
    <color name="colorPrimaryDark">#0f0f23</color>
    <color name="colorAccent">#8b5cf6</color>
    <color name="splashBackground">#0f0f23</color>
</resources>`
};

// Create configuration files
Object.keys(configFiles).forEach(filePath => {
    const fullPath = path.join(__dirname, '..', filePath);
    const dir = path.dirname(fullPath);
    
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    
    fs.writeFileSync(fullPath, configFiles[filePath]);
});

console.log('📋 Created Android configuration files');

// 5. Generate app icons and splash screens
generateAppAssets();

function generateAppAssets() {
    console.log('🎨 Generating app icons and splash screens...');
    
    // Create basic SVG icon
    const iconSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192" width="192" height="192">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#6366f1;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="192" height="192" rx="24" fill="url(#grad)"/>
        <path d="M96 40L104.18 82.52L160 90L104.18 97.48L96 140L87.82 97.48L32 90L87.82 82.52L96 40Z" fill="white"/>
        <path d="M152 120L155.5 137L176 140L155.5 143L152 160L148.5 143L128 140L148.5 137L152 120Z" fill="white"/>
        <path d="M40 50L42 60L56 62L42 64L40 74L38 64L24 62L38 60L40 50Z" fill="white"/>
        <text x="96" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="white">YOUSEF SH</text>
    </svg>`;
    
    // Create splash screen SVG
    const splashSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1920" width="1080" height="1920">
        <defs>
            <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#0f0f23;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#1a1a2e;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="iconGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#6366f1;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="1080" height="1920" fill="url(#bg)"/>
        <g transform="translate(540,960)">
            <rect x="-80" y="-80" width="160" height="160" rx="20" fill="url(#iconGrad)"/>
            <path d="M0 -40L8.18 2.52L80 10L8.18 17.48L0 60L-8.18 17.48L-80 10L-8.18 2.52L0 -40Z" fill="white"/>
            <path d="M40 -20L43.5 -3L60 0L43.5 3L40 20L36.5 3L20 0L36.5 -3L40 -20Z" fill="white"/>
            <path d="-40 -30L-38 -20L-24 -18L-38 -16L-40 -6L-42 -16L-56 -18L-42 -20L-40 -30Z" fill="white"/>
        </g>
        <text x="540" y="1100" text-anchor="middle" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="white">YOUSEF SH</text>
        <text x="540" y="1150" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#94a3b8">المساعد الذكي</text>
    </svg>`;
    
    // Save SVG files
    const assetsDir = path.join(__dirname, '..', 'cordova', 'res', 'android');
    const iconsDir = path.join(assetsDir, 'icons');
    const splashDir = path.join(assetsDir, 'splash');
    
    [iconsDir, splashDir].forEach(dir => {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    });
    
    fs.writeFileSync(path.join(iconsDir, 'icon.svg'), iconSVG);
    fs.writeFileSync(path.join(splashDir, 'splash.svg'), splashSVG);
    
    // Create placeholder PNG files (in production, use proper image generation)
    const iconSizes = [36, 48, 72, 96, 144, 192];
    const splashSizes = [
        { name: 'splash-200x320.png', width: 200, height: 320 },
        { name: 'splash-320x480.png', width: 320, height: 480 },
        { name: 'splash-480x800.png', width: 480, height: 800 },
        { name: 'splash-720x1280.png', width: 720, height: 1280 },
        { name: 'splash-960x1600.png', width: 960, height: 1600 },
        { name: 'splash-1280x1920.png', width: 1280, height: 1920 },
        { name: 'splash-320x200.png', width: 320, height: 200 },
        { name: 'splash-480x320.png', width: 480, height: 320 },
        { name: 'splash-800x480.png', width: 800, height: 480 },
        { name: 'splash-1280x720.png', width: 1280, height: 720 },
        { name: 'splash-1600x960.png', width: 1600, height: 960 },
        { name: 'splash-1920x1280.png', width: 1920, height: 1280 }
    ];
    
    console.log('📝 Generated SVG assets (PNG conversion needed for production)');
}

console.log('✅ Mobile optimization completed!');
console.log('📱 Ready to build Android APK');