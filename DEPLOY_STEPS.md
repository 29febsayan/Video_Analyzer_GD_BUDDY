# 🚀 Step-by-Step Deployment Guide

Follow these steps to deploy your Visual Behavior Analysis API to production.

## ✅ Pre-Deployment Checklist

All files are ready! ✅
- ✅ Dockerfile
- ✅ main.py
- ✅ requirements.txt
- ✅ Platform configs (Railway, Render)
- ✅ API key system
- ✅ Health checks

---

## 📝 Step 1: Prepare GitHub Repository

### 1.1 Create GitHub Repository

1. Go to: **https://github.com/new**
2. Repository name: `video-analysis-api` (or any name)
3. Description: "Visual Behavior Analysis API"
4. Choose: **Public** or **Private**
5. **IMPORTANT**: Do NOT check "Initialize with README"
6. Click **"Create repository"**

### 1.2 Copy Repository URL

After creating, you'll see a page with setup instructions. Copy the URL that looks like:
```
https://github.com/YOUR_USERNAME/video-analysis-api.git
```

---

## 📤 Step 2: Push Code to GitHub

Open PowerShell or Command Prompt in your project folder and run:

```powershell
cd "C:\Users\91947\Desktop\video analysis"

# Check git status
git status

# Add all files
git add .

# Commit
git commit -m "Initial deployment - Visual Behavior Analysis API"

# Add GitHub remote (replace with YOUR repository URL)
git remote add origin https://github.com/YOUR_USERNAME/video-analysis-api.git

# Push to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

---

## 🚂 Step 3: Deploy on Railway (Recommended - Easiest)

### 3.1 Sign Up / Login

1. Go to: **https://railway.app**
2. Click **"Login"** or **"Start a New Project"**
3. Sign up with **GitHub** (easiest option)
4. Authorize Railway to access your GitHub

### 3.2 Deploy Your Project

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Authorize Railway (if not already done)
4. Select your repository: **video-analysis-api**
5. Railway will automatically:
   - Detect the Dockerfile
   - Build your container
   - Deploy your API

### 3.3 Get Your API URL

1. Wait for deployment to complete (2-3 minutes)
2. Click on your project
3. Go to **"Settings"** tab
4. Scroll to **"Domains"**
5. Click **"Generate Domain"**
6. Your API URL will be: `https://your-app-name.up.railway.app`

### 3.4 Generate API Key

**Option A: Via Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to project
railway link

# Generate key
railway run python api_keys.py
```

**Option B: Via API Endpoint**
```bash
curl -X POST "https://your-app-name.up.railway.app/api-key/generate" \
  -H "Content-Type: application/json" \
  -d '{"name": "production_key", "expires_days": null}'
```

**Option C: Via Railway Dashboard**
1. Go to your project → **"Variables"** tab
2. Add variable if needed
3. Use Railway's shell feature to run: `python api_keys.py`

---

## 🎨 Alternative: Deploy on Render

### 3.1 Sign Up / Login

1. Go to: **https://render.com**
2. Click **"Get Started"**
3. Sign up with **GitHub**

### 3.2 Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account (if not already)
3. Select repository: **video-analysis-api**

### 3.3 Configure Service

- **Name**: `visual-behavior-analysis-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Plan**: **Free** (or Starter for better performance)

### 3.4 Deploy

1. Click **"Create Web Service"**
2. Wait for build and deployment (3-5 minutes)
3. Your API will be at: `https://visual-behavior-analysis-api.onrender.com`

### 3.5 Generate API Key

**Option A: Via Render Shell**
1. Go to your service → **"Shell"** tab
2. Run: `python api_keys.py`
3. Copy the generated key

**Option B: Via API Endpoint**
```bash
curl -X POST "https://visual-behavior-analysis-api.onrender.com/api-key/generate" \
  -H "Content-Type: application/json" \
  -d '{"name": "production_key"}'
```

---

## ✅ Step 4: Test Your Deployment

### 4.1 Health Check

```bash
curl https://your-api-url.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "analyzer_ready": true,
  "timestamp": 1704067200.0
}
```

### 4.2 View API Documentation

Open in browser:
```
https://your-api-url.com/docs
```

### 4.3 Test Analysis Endpoint

```bash
# Replace with your actual API key and URL
API_KEY="your_generated_api_key"
API_URL="https://your-api-url.com"

curl -X POST "$API_URL/analyze/frame" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test_frame.jpg"
```

---

## 🔑 Step 5: Save Your API Key

**IMPORTANT**: Save your API key securely! You'll need it for all API requests.

Create a file `API_KEY.txt` (or save in a password manager):

```
API URL: https://your-api-url.com
API Key: your_generated_api_key_here
```

---

## 📊 Your Deployment is Complete!

You now have:
- ✅ Production API URL
- ✅ HTTPS enabled
- ✅ API key for authentication
- ✅ Auto-scaling
- ✅ Monitoring & logs
- ✅ API documentation

---

## 🎯 Quick Reference

### Your API Endpoints

- **Health**: `GET https://your-api-url.com/health`
- **Analyze Frame**: `POST https://your-api-url.com/analyze/frame`
- **Analyze Base64**: `POST https://your-api-url.com/analyze/base64`
- **API Docs**: `https://your-api-url.com/docs`

### Using Your API

```python
import requests

API_KEY = "your_api_key_here"
API_URL = "https://your-api-url.com"

# Analyze frame
with open("frame.jpg", "rb") as f:
    response = requests.post(
        f"{API_URL}/analyze/frame",
        files={"file": f},
        headers={"X-API-Key": API_KEY}
    )
    result = response.json()
    print(f"Attention: {result['attention']}%")
```

---

## 🆘 Troubleshooting

### Git Push Issues

If you get authentication errors:
1. Use GitHub Personal Access Token instead of password
2. Or use SSH: `git remote set-url origin git@github.com:USERNAME/REPO.git`

### Deployment Fails

1. Check platform logs for errors
2. Verify Dockerfile is correct
3. Ensure all files are committed
4. Check requirements.txt

### API Not Responding

1. Check health endpoint first
2. Verify API key is correct
3. Check platform logs
4. Ensure service is running (not sleeping on free tier)

---

## 📞 Need Help?

- **Railway Support**: https://docs.railway.app
- **Render Support**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

**🎉 Congratulations! Your API is now live in production!**

