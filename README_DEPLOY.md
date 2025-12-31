# 🚀 DEPLOY NOW - Quick Instructions

## ⚡ Fastest Way to Deploy (5 minutes)

### 1. Create GitHub Repo
- Go to: https://github.com/new
- Name: `video-analysis-api`
- **Don't** initialize with README
- Copy the repository URL

### 2. Push Code
```powershell
cd "C:\Users\91947\Desktop\video analysis"
git add .
git commit -m "Deploy Visual Behavior Analysis API"
git remote add origin https://github.com/YOUR_USERNAME/video-analysis-api.git
git push -u origin main
```

### 3. Deploy on Railway
- Go to: https://railway.app
- Sign up with GitHub
- Click "New Project" → "Deploy from GitHub"
- Select your repo
- **Done!** Railway auto-deploys

### 4. Get API Key
After deployment, visit:
```
https://your-app.up.railway.app/api-key/generate
```

Or use Railway CLI:
```bash
railway run python api_keys.py
```

---

## 📋 Full Instructions

See `DEPLOY_STEPS.md` for detailed step-by-step guide.

---

## ✅ What You'll Get

- Production API URL (HTTPS)
- API key for authentication
- Auto-scaling
- Monitoring & logs
- API documentation at `/docs`

---

**That's it! Your API will be live in ~5 minutes! 🎉**

