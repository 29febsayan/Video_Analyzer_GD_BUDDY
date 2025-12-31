# ⚡ Quick Deploy Guide

## 🚂 Railway (Fastest - 5 minutes)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Deploy Visual Behavior Analysis API"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Railway auto-detects Dockerfile and deploys!

3. **Get API Key**
   - After deployment, visit: `https://your-app.up.railway.app/docs`
   - Use `/api-key/generate` endpoint to create API key
   - Or SSH: `railway run python api_keys.py`

**Done!** Your API is live at: `https://your-app.up.railway.app`

---

## 🎨 Render (Easy - 5 minutes)

1. **Push to GitHub** (same as above)

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect GitHub → Select repo
   - Settings:
     - Build: `pip install -r requirements.txt`
     - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

3. **Get API Key**
   - After deployment, use Render Shell
   - Or use `/api-key/generate` endpoint

**Done!** Your API is live at: `https://your-app.onrender.com`

---

## 🔑 Your API Key

After deployment, generate your API key:

```bash
# Via API
curl -X POST "https://your-api-url.com/api-key/generate" \
  -H "Content-Type: application/json" \
  -d '{"name": "production_key"}'
```

Save the returned API key - you'll need it for all requests!

---

## ✅ Test Your Deployment

```bash
# Health check
curl https://your-api-url.com/health

# Test analysis (replace API_KEY)
curl -X POST "https://your-api-url.com/analyze/frame" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@test_frame.jpg"
```

---

## 📚 Full Documentation

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

