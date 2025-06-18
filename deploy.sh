#!/bin/bash

# TDS Virtual TA Deployment Script
echo "🚀 Deploying TDS Virtual TA to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI is not installed. Please install it first:"
    echo "npm i -g vercel"
    exit 1
fi

# Check if user is logged in
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel first:"
    vercel login
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Make sure to set AIPROXY_TOKEN in Vercel dashboard."
    echo "You can create a .env file with: echo 'AIPROXY_TOKEN=your_api_key' > .env"
fi

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "🌐 Your app should be live at the URL shown above"
echo "🔧 Don't forget to set AIPROXY_TOKEN in your Vercel dashboard if you haven't already" 