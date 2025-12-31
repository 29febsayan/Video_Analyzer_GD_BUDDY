# 🚀 Production Deployment Guide

This guide covers deploying the Visual Behavior Analysis API to production platforms.

## 📋 Prerequisites

1. **GitHub Account** (for code repository)
2. **Platform Account** (Railway, Render, or Fly.io)
3. **API Key** (generated during deployment)

---

## 🚂 Option 1: Railway Deployment (Recommended)

Railway is easy to use and offers a free tier.

### Step 1: Prepare Repository

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit - Visual Behavior Analysis API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/video-analysis-api.git
git push -u origin main
```

### Step 2: Deploy on Railway

1. **Sign up/Login**: Go to [railway.app](https://railway.app) and sign up
2. **New Project**: Click "New Project"
3. **Deploy from GitHub**: Select "Deploy from GitHub repo"
4. **Select Repository**: Choose your repository
5. **Auto-Deploy**: Railway will automatically detect the Dockerfile and deploy

### Step 3: Configure Environment Variables

In Railway dashboard:
1. Go to your project → Variables
2. Add these variables (if needed):
   - `PORT` (auto-set by Railway)
   - `ENV=production`

### Step 4: Generate API Key

After deployment, SSH into the container or use Railway CLI:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to project
railway link

# Run command to generate API key
railway run python api_keys.py
```

Or access the deployed API and use the `/api-key/generate` endpoint.

### Step 5: Get Your API URL

Railway will provide a URL like: `https://your-app-name.up.railway.app`

**Your API will be available at:**
- Base URL: `https://your-app-name.up.railway.app`
- Docs: `https://your-app-name.up.railway.app/docs`
- Health: `https://your-app-name.up.railway.app/health`

---

## 🎨 Option 2: Render Deployment

Render offers free tier with automatic HTTPS.

### Step 1: Prepare Repository

Same as Railway - push to GitHub.

### Step 2: Deploy on Render

1. **Sign up/Login**: Go to [render.com](https://render.com) and sign up
2. **New Web Service**: Click "New +" → "Web Service"
3. **Connect GitHub**: Connect your GitHub account
4. **Select Repository**: Choose your repository
5. **Configure**:
   - **Name**: `visual-behavior-analysis-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or Starter for better performance)

### Step 3: Environment Variables

In Render dashboard → Environment:
- `PORT` (auto-set)
- `ENV=production`

### Step 4: Generate API Key

After deployment, use the shell feature or API endpoint:

```bash
# Via Render Shell
render shell
python api_keys.py
```

### Step 5: Get Your API URL

Render provides: `https://visual-behavior-analysis-api.onrender.com`

---

## 🐳 Option 3: Docker Deployment (Any Platform)

### Build Docker Image

```bash
docker build -t visual-behavior-analysis-api .
```

### Run Locally

```bash
docker run -p 8000:8000 -e PORT=8000 visual-behavior-analysis-api
```

### Push to Docker Hub

```bash
docker tag visual-behavior-analysis-api YOUR_USERNAME/visual-behavior-analysis-api
docker push YOUR_USERNAME/visual-behavior-analysis-api
```

Then deploy to any platform that supports Docker (Fly.io, DigitalOcean, AWS, etc.)

---

## 🔐 Setting Up API Keys in Production

### Method 1: Via API Endpoint

After deployment, generate API key:

```bash
curl -X POST "https://your-api-url.com/api-key/generate" \
  -H "Content-Type: application/json" \
  -d '{"name": "production_key", "expires_days": null}'
```

**⚠️ SECURITY**: In production, protect the `/api-key/generate` endpoint with admin authentication!

### Method 2: Via Environment Variable

1. Generate API key locally: `python api_keys.py`
2. Add to platform environment variables:
   - `DEFAULT_API_KEY=your_generated_key_here`
3. Modify `main.py` to load from environment if needed

### Method 3: Pre-generate and Store

1. Generate API key: `python api_keys.py`
2. Add `api_keys.json` to your repository (or use secrets management)
3. Deploy with the key file included

---

## 🔒 Security Best Practices

### 1. Protect API Key Generation

Add authentication to `/api-key/generate`:

```python
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

@app.post("/api-key/generate")
async def generate_api_key(
    password: str = Header(None),
    name: str = "default",
    expires_days: Optional[int] = None
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")
    # ... rest of code
```

### 2. Use Environment Variables

Never commit sensitive data:
- Add `.env` to `.gitignore`
- Use platform secrets management
- Store API keys in secure vaults

### 3. Enable HTTPS

All platforms provide HTTPS automatically. Always use HTTPS in production.

### 4. Rate Limiting

Consider adding rate limiting:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze/frame")
@limiter.limit("10/minute")
async def analyze_frame(...):
    # ...
```

---

## 📊 Monitoring & Logs

### Railway

- View logs in Railway dashboard
- Set up alerts for errors
- Monitor resource usage

### Render

- View logs in Render dashboard
- Set up health checks
- Monitor uptime

### Custom Monitoring

Add logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/analyze/frame")
async def analyze_frame(...):
    logger.info(f"Processing frame from {request.client.host}")
    # ...
```

---

## 🧪 Testing Production Deployment

### 1. Health Check

```bash
curl https://your-api-url.com/health
```

### 2. Test Analysis Endpoint

```bash
API_KEY="your_production_api_key"
curl -X POST "https://your-api-url.com/analyze/frame" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test_frame.jpg"
```

### 3. Check Documentation

Visit: `https://your-api-url.com/docs`

---

## 🔄 Continuous Deployment

### Railway

- Automatic deployment on git push
- Branch deployments available
- Rollback support

### Render

- Automatic deployment on git push
- Manual deployments available
- Blue-green deployments

---

## 💰 Cost Estimation

### Free Tiers

- **Railway**: $5 free credit/month (usually enough for testing)
- **Render**: Free tier with limitations (spins down after inactivity)
- **Fly.io**: Free tier available

### Paid Tiers

- **Railway**: Pay-as-you-go, ~$5-20/month for moderate usage
- **Render**: Starter plan ~$7/month
- **Fly.io**: Pay-as-you-go

---

## 🐛 Troubleshooting

### Build Fails

1. Check Dockerfile syntax
2. Verify all dependencies in requirements.txt
3. Check platform logs for errors

### API Not Responding

1. Check health endpoint: `/health`
2. Verify PORT environment variable
3. Check platform logs
4. Verify API key is correct

### Memory Issues

1. Increase platform memory allocation
2. Optimize image processing
3. Add resource limits

---

## 📝 Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Dockerfile created and tested
- [ ] Environment variables configured
- [ ] API key generated
- [ ] Health check working
- [ ] API endpoints tested
- [ ] Documentation accessible
- [ ] HTTPS enabled
- [ ] Monitoring set up
- [ ] Error handling verified

---

## 🎯 Quick Deploy Commands

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up
```

### Render

Just connect GitHub repo and Render handles the rest!

---

## 📞 Support

- **Railway Docs**: https://docs.railway.app
- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

**Your API is ready for production deployment! 🚀**

Choose a platform and follow the steps above. Railway is recommended for ease of use.

