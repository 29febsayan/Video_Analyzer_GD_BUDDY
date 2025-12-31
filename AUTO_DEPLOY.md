# 🚀 Automated Deployment Guide

This guide will help you deploy your Visual Behavior Analysis API in just a few minutes!

## ⚡ Quick Start (3 Steps)

### Step 1: Run the Deployment Helper

```bash
python deploy_now.py
```

This script will:
- ✅ Check all required files
- ✅ Initialize Git repository
- ✅ Guide you through GitHub setup
- ✅ Push code to GitHub
- ✅ Provide platform-specific instructions

### Step 2: Create GitHub Repository

The script will guide you, or manually:
1. Go to https://github.com/new
2. Create a new repository (name: `video-analysis-api`)
3. **DO NOT** initialize with README/license
4. Copy the repository URL

### Step 3: Deploy on Platform

Choose one:

#### Option A: Railway (Easiest - 2 minutes)

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub"
4. Select your repository
5. **Done!** Railway auto-deploys

#### Option B: Render (Free Tier - 3 minutes)

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect GitHub → Select repo
5. Settings:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"

---

## 📋 What Happens During Deployment

1. **Git Setup**: Code is pushed to GitHub
2. **Platform Build**: Platform builds Docker container
3. **Dependencies**: All Python packages installed
4. **API Starts**: FastAPI server starts automatically
5. **HTTPS**: Platform provides HTTPS automatically
6. **Domain**: Platform provides public URL

---

## 🔑 After Deployment - Get Your API Key

### Method 1: Via API Endpoint

```bash
curl -X POST "https://your-api-url.com/api-key/generate" \
  -H "Content-Type: application/json" \
  -d '{"name": "production_key"}'
```

### Method 2: Via Platform Shell

**Railway:**
```bash
railway run python api_keys.py
```

**Render:**
- Use Render Shell → Run: `python api_keys.py`

---

## ✅ Test Your Deployment

### 1. Health Check

```bash
curl https://your-api-url.com/health
```

### 2. View Documentation

Visit: `https://your-api-url.com/docs`

### 3. Test Analysis

```bash
API_KEY="your_generated_api_key"
curl -X POST "https://your-api-url.com/analyze/frame" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test_frame.jpg"
```

---

## 🎯 Complete Deployment Checklist

- [ ] Run `python deploy_now.py`
- [ ] Create GitHub repository
- [ ] Code pushed to GitHub
- [ ] Choose platform (Railway or Render)
- [ ] Deploy on platform
- [ ] Get deployment URL
- [ ] Generate API key
- [ ] Test health endpoint
- [ ] Test analysis endpoint
- [ ] Save API key securely

---

## 🆘 Troubleshooting

### Git Push Fails

If git push fails, manually run:
```bash
git remote add origin YOUR_GITHUB_URL
git add .
git commit -m "Initial deployment"
git push -u origin main
```

### Platform Build Fails

1. Check platform logs for errors
2. Verify Dockerfile is correct
3. Check requirements.txt has all dependencies
4. Ensure main.py exists

### API Not Responding

1. Check health endpoint: `/health`
2. Verify PORT environment variable is set
3. Check platform logs
4. Ensure API key is correct

---

## 📞 Need Help?

- **Railway Docs**: https://docs.railway.app
- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

## 🎉 Success!

Once deployed, you'll have:
- ✅ Production API URL
- ✅ HTTPS enabled
- ✅ Auto-scaling
- ✅ Monitoring & logs
- ✅ API documentation

**Your API is ready to use in production! 🚀**

