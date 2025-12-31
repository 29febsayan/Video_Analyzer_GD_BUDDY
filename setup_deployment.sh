#!/bin/bash
# Setup script for production deployment

echo "=========================================="
echo "Visual Behavior Analysis API - Deployment Setup"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git branch -M main
    echo "[OK] Git repository initialized"
else
    echo "[OK] Git repository already exists"
fi

# Check if .gitignore exists
if [ ! -f ".gitignore" ]; then
    echo "[WARNING] .gitignore not found - creating one..."
    # .gitignore should already exist, but just in case
fi

# Check Dockerfile
if [ ! -f "Dockerfile" ]; then
    echo "[ERROR] Dockerfile not found!"
    exit 1
else
    echo "[OK] Dockerfile found"
fi

# Check requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found!"
    exit 1
else
    echo "[OK] requirements.txt found"
fi

# Check main.py
if [ ! -f "main.py" ]; then
    echo "[ERROR] main.py not found!"
    exit 1
else
    echo "[OK] main.py found"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository"
echo "2. Add remote: git remote add origin YOUR_REPO_URL"
echo "3. Commit and push:"
echo "   git add ."
echo "   git commit -m 'Initial deployment setup'"
echo "   git push -u origin main"
echo "4. Deploy on Railway or Render (see QUICK_DEPLOY.md)"
echo ""

