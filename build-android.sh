#!/bin/bash

echo "🚀 بناء تطبيق YOUSEF SH AI Agent للأندرويد"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js غير مثبت. يرجى تثبيت Node.js أولاً"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm غير مثبت. يرجى تثبيت npm أولاً"
    exit 1
fi

print_status "1/8 التحقق من البيئة..."

# Create cordova directory if it doesn't exist
if [ ! -d "cordova" ]; then
    print_status "2/8 إنشاء مجلد Cordova..."
    mkdir -p cordova
fi

# Install Cordova globally if not installed
if ! command -v cordova &> /dev/null; then
    print_status "تثبيت Cordova عالمياً..."
    npm install -g cordova
fi

# Create Cordova project
print_status "3/8 إنشاء مشروع Cordova..."
cd cordova

# Initialize Cordova project if not already initialized
if [ ! -f "config.xml" ]; then
    cordova create . com.yousefsh.aiagent "YOUSEF SH" --template=blank
    print_success "تم إنشاء مشروع Cordova"
else
    print_success "مشروع Cordova موجود بالفعل"
fi

# Copy config.xml from parent directory
if [ -f "../config.xml" ]; then
    cp "../config.xml" "./config.xml"
    print_success "تم نسخ ملف config.xml"
fi

# Add Android platform
print_status "4/8 إضافة منصة الأندرويد..."
if ! cordova platform list | grep -q "android"; then
    cordova platform add android
    print_success "تم إضافة منصة الأندرويد"
else
    print_success "منصة الأندرويد موجودة بالفعل"
fi

# Copy mobile HTML file to www
print_status "5/8 نسخ ملفات التطبيق..."
if [ -f "../mobile/index.html" ]; then
    cp "../mobile/index.html" "./www/index.html"
    print_success "تم نسخ الملف الرئيسي للهاتف المحمول"
else
    print_error "ملف mobile/index.html غير موجود"
    exit 1
fi

# Create basic icons (placeholder)
print_status "6/8 إنشاء الأيقونات..."
mkdir -p res/android/icons
mkdir -p res/android/splash

# Create simple colored rectangles as placeholders for icons
# In production, you would use proper image generation tools
echo "Creating placeholder icons..."

# Create basic icon files (these are just placeholder files)
# In a real scenario, you'd use imagemagick or similar tools
for size in 36 48 72 96 144 192; do
    # Create placeholder PNG files
    touch "res/android/icons/icon-${size}.png"
done

# Create splash screen files
splash_sizes=("200x320" "320x480" "480x800" "720x1280" "960x1600" "1280x1920" "320x200" "480x320" "800x480" "1280x720" "1600x960" "1920x1280")
for size in "${splash_sizes[@]}"; do
    touch "res/android/splash/splash-${size}.png"
done

print_success "تم إنشاء ملفات الأيقونات المؤقتة"

# Install required plugins
print_status "7/8 تثبيت الإضافات المطلوبة..."

plugins=(
    "cordova-plugin-whitelist"
    "cordova-plugin-device"
    "cordova-plugin-network-information"
    "cordova-plugin-file"
    "cordova-plugin-statusbar"
    "cordova-plugin-splashscreen"
)

for plugin in "${plugins[@]}"; do
    if ! cordova plugin list | grep -q "$plugin"; then
        print_status "تثبيت $plugin..."
        cordova plugin add $plugin
    else
        print_success "$plugin مثبت بالفعل"
    fi
done

# Build the Android APK
print_status "8/8 بناء ملف APK..."

# Clean previous builds
cordova clean android

# Build for debug
print_status "بناء النسخة التجريبية..."
if cordova build android; then
    print_success "تم بناء النسخة التجريبية بنجاح!"
    
    # Find the generated APK
    APK_PATH=$(find platforms/android/app/build/outputs/apk -name "*.apk" -type f | head -1)
    
    if [ -n "$APK_PATH" ]; then
        # Copy APK to project root with a descriptive name
        cp "$APK_PATH" "../yousef-sh-debug.apk"
        print_success "تم نسخ ملف APK إلى: yousef-sh-debug.apk"
        
        # Get APK size
        APK_SIZE=$(du -h "../yousef-sh-debug.apk" | cut -f1)
        print_success "حجم الملف: $APK_SIZE"
        
        echo ""
        echo "🎉 تم بناء التطبيق بنجاح!"
        echo "📱 ملف APK: yousef-sh-debug.apk"
        echo "📏 الحجم: $APK_SIZE"
        echo ""
        echo "لتثبيت التطبيق على جهاز الأندرويد:"
        echo "1. انقل ملف yousef-sh-debug.apk إلى جهازك"
        echo "2. فعّل 'المصادر غير المعروفة' في إعدادات الأمان"
        echo "3. اضغط على الملف لتثبيته"
        echo ""
        
    else
        print_error "لم يتم العثور على ملف APK"
        exit 1
    fi
else
    print_error "فشل في بناء التطبيق"
    echo ""
    echo "نصائح لحل المشاكل:"
    echo "1. تأكد من تثبيت Android SDK"
    echo "2. تأكد من تثبيت Gradle"
    echo "3. تأكد من إعداد متغيرات البيئة ANDROID_HOME و JAVA_HOME"
    echo "4. جرب تشغيل: cordova requirements android"
    exit 1
fi

# Optional: Build release version
read -p "هل تريد بناء النسخة النهائية (release)؟ (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "بناء النسخة النهائية..."
    print_warning "ستحتاج إلى keystore للتوقيع"
    
    if cordova build android --release; then
        print_success "تم بناء النسخة النهائية!"
        
        RELEASE_APK=$(find platforms/android/app/build/outputs/apk -name "*release*.apk" -type f | head -1)
        if [ -n "$RELEASE_APK" ]; then
            cp "$RELEASE_APK" "../yousef-sh-release-unsigned.apk"
            print_success "تم نسخ النسخة النهائية: yousef-sh-release-unsigned.apk"
            print_warning "النسخة النهائية غير موقعة. ستحتاج لتوقيعها قبل نشرها."
        fi
    else
        print_warning "فشل في بناء النسخة النهائية"
    fi
fi

cd ..

print_success "انتهى بناء التطبيق!"
echo ""
echo "🔧 المتطلبات للتطوير المستقبلي:"
echo "• Android Studio للتطوير المتقدم"
echo "• Keystore للنسخ النهائية الموقعة"
echo "• إضافات Cordova إضافية حسب الحاجة"
echo ""
echo "📚 مصادر مفيدة:"
echo "• دليل Cordova: https://cordova.apache.org/docs/"
echo "• دليل Android: https://developer.android.com/"
echo ""
echo "شكراً لاستخدام YOUSEF SH AI Agent! 🚀"