#!/usr/bin/env python3
"""
Simple test script to verify the local setup works correctly.
Run this before deploying to Vercel.
"""

import os
import sys
import asyncio
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if environment variables are set correctly."""
    print("🔍 Testing environment variables...")
    
    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("❌ AIPROXY_TOKEN not found in environment variables")
        print("   Please create a .env file with: AIPROXY_TOKEN=your_api_key")
        return False
    
    print("✅ AIPROXY_TOKEN found")
    return True

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("📦 Testing dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'aiohttp', 
        'python-dotenv', 'numpy', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📋 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_database():
    """Test if the database exists and is accessible."""
    print("🗄️  Testing database...")
    
    db_path = "knowledge_base.db"
    if not os.path.exists(db_path):
        print("❌ knowledge_base.db not found")
        print("   Make sure you have scraped data and created the database")
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'discourse_chunks' in tables or 'markdown_chunks' in tables:
            print("✅ Database tables found")
            conn.close()
            return True
        else:
            print("❌ No knowledge base tables found")
            conn.close()
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_api_endpoint():
    """Test if the API endpoint responds correctly."""
    print("🌐 Testing API endpoint...")
    
    try:
        # Start the server in a subprocess
        import subprocess
        import time
        
        # Start the server
        process = subprocess.Popen([sys.executable, "app.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ API health endpoint working")
            
            # Test query endpoint
            test_data = {"question": "What is machine learning?"}
            response = requests.post("http://localhost:8000/query", 
                                   json=test_data, 
                                   timeout=30)
            
            if response.status_code == 200:
                print("✅ API query endpoint working")
                result = response.json()
                if "answer" in result:
                    print(f"✅ Got response: {result['answer'][:100]}...")
                else:
                    print("❌ Response missing 'answer' field")
                    return False
            else:
                print(f"❌ Query endpoint failed: {response.status_code}")
                return False
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Stop the server
        process.terminate()
        process.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_frontend():
    """Test if frontend files exist."""
    print("🎨 Testing frontend files...")
    
    required_files = ['index.html', 'styles.css', 'script.js']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n📋 Missing frontend files: {', '.join(missing_files)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 TDS Virtual TA - Local Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment),
        ("Dependencies", test_dependencies),
        ("Database", test_database),
        ("Frontend Files", test_frontend),
        ("API Endpoint", test_api_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready for deployment.")
        print("\n🚀 To deploy to Vercel:")
        print("   1. Run: vercel --prod")
        print("   2. Set AIPROXY_TOKEN in Vercel dashboard")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 