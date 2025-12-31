# 🚨 CRITICAL: Railway StartCommand Override Fix

## ⚠️ The Problem

Railway is using a **startCommand** from the dashboard that's overriding your Dockerfile CMD. This is why `${PORT:-8000}` is being passed as a literal string instead of being expanded.

## ✅ Solution: Remove StartCommand in Railway Dashboard

### Step 1: Go to Railway Dashboard

1. Open: https://railway.app
2. Click on your project
3. Click on your service

### Step 2: Remove StartCommand

1. Go to **Settings** tab
2. Scroll down to **"Deploy"** section
3. Find **"Start Command"** field
4. **DELETE** or **CLEAR** everything in that field
5. Leave it **COMPLETELY EMPTY**
6. Click **"Save"** or **"Update"**

### Step 3: Verify

- The Start Command field should be **empty**
- Railway will now use the Dockerfile CMD: `["/bin/sh", "/app/start.sh"]`

## 🔧 Alternative: If You Can't Access Dashboard

If you can't access the dashboard, we can modify the approach to work even with a startCommand override.

### Option A: Use Python to read PORT

Create a Python startup script instead:

```python
# start.py
import os
import sys

port = int(os.environ.get("PORT", 8000))
os.execvp("uvicorn", ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)])
```

But first, try removing the startCommand in the dashboard - that's the proper fix!

## 📤 After Removing StartCommand

1. Railway will automatically redeploy
2. Or trigger a manual redeploy
3. The start.sh script will execute properly
4. PORT variable will be expanded correctly

## ✅ Expected Result

After removing the startCommand:
- ✅ start.sh will execute
- ✅ PORT will be read from environment
- ✅ uvicorn will start with correct port
- ✅ No more "Invalid value for --port" errors

---

**This is the most common issue with Railway deployments! The dashboard startCommand overrides everything.**

