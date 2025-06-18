@echo off
echo 🚀 Deploying TDS Virtual TA to Vercel...

REM Check if Vercel CLI is installed
vercel --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Vercel CLI is not installed. Please install it first:
    echo npm i -g vercel
    pause
    exit /b 1
)

REM Check if user is logged in
vercel whoami >nul 2>&1
if errorlevel 1 (
    echo 🔐 Please login to Vercel first:
    vercel login
)

REM Check if .env file exists
if not exist .env (
    echo ⚠️  Warning: .env file not found. Make sure to set AIPROXY_TOKEN in Vercel dashboard.
    echo You can create a .env file with: echo AIPROXY_TOKEN=your_api_key > .env
)

REM Deploy to Vercel
echo 📦 Deploying to Vercel...
vercel --prod

echo ✅ Deployment complete!
echo 🌐 Your app should be live at the URL shown above
echo 🔧 Don't forget to set AIPROXY_TOKEN in your Vercel dashboard if you haven't already
pause 