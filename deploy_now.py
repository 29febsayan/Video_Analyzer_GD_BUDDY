#!/usr/bin/env python3
"""
Automated Deployment Helper

This script helps you deploy the Visual Behavior Analysis API
to Railway or Render with minimal manual steps.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git():
    """Check if git is initialized and configured."""
    try:
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Git repository initialized")
            return True
        else:
            print("[INFO] Git repository not initialized")
            return False
    except FileNotFoundError:
        print("[ERROR] Git is not installed. Please install Git first.")
        return False

def init_git():
    """Initialize git repository."""
    print("\nInitializing Git repository...")
    try:
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'branch', '-M', 'main'], check=True)
        print("[OK] Git repository initialized")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to initialize git: {e}")
        return False

def check_files():
    """Check if all required files exist."""
    required_files = [
        'Dockerfile',
        'main.py',
        'requirements.txt',
        'production_ready_analyzer.py',
        'api_keys.py'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
        else:
            print(f"[OK] {file}")
    
    if missing:
        print(f"\n[ERROR] Missing required files: {', '.join(missing)}")
        return False
    
    return True

def create_github_repo_instructions():
    """Print instructions for creating GitHub repository."""
    print("\n" + "=" * 70)
    print("STEP 1: Create GitHub Repository")
    print("=" * 70)
    print("\n1. Go to: https://github.com/new")
    print("2. Repository name: video-analysis-api (or any name you prefer)")
    print("3. Make it Public or Private (your choice)")
    print("4. DO NOT initialize with README, .gitignore, or license")
    print("5. Click 'Create repository'")
    print("\nAfter creating, you'll get a URL like:")
    print("   https://github.com/YOUR_USERNAME/video-analysis-api.git")
    print("\nCopy that URL - you'll need it in the next step!")
    print("=" * 70)

def prepare_deployment():
    """Prepare all files for deployment."""
    print("\n" + "=" * 70)
    print("PREPARING DEPLOYMENT")
    print("=" * 70)
    
    # Check files
    print("\n1. Checking required files...")
    if not check_files():
        return False
    
    # Check/Initialize git
    print("\n2. Checking Git...")
    if not check_git():
        if not init_git():
            return False
    
    # Check if .gitignore exists
    if not Path('.gitignore').exists():
        print("[WARNING] .gitignore not found, but continuing...")
    
    print("\n[SUCCESS] All files are ready for deployment!")
    return True

def railway_deploy_instructions():
    """Print Railway deployment instructions."""
    print("\n" + "=" * 70)
    print("RAILWAY DEPLOYMENT (Recommended - Easiest)")
    print("=" * 70)
    print("\n1. Go to: https://railway.app")
    print("2. Sign up/Login (you can use GitHub to sign in)")
    print("3. Click 'New Project'")
    print("4. Select 'Deploy from GitHub repo'")
    print("5. Authorize Railway to access your GitHub")
    print("6. Select your repository: video-analysis-api")
    print("7. Railway will automatically:")
    print("   - Detect the Dockerfile")
    print("   - Build the container")
    print("   - Deploy your API")
    print("\n8. After deployment:")
    print("   - Click on your project")
    print("   - Go to 'Settings' → 'Generate Domain'")
    print("   - Your API will be at: https://your-app.up.railway.app")
    print("\n9. Generate API Key:")
    print("   - Click 'Variables' tab")
    print("   - Or use: railway run python api_keys.py")
    print("   - Or visit: https://your-app.up.railway.app/api-key/generate")
    print("=" * 70)

def render_deploy_instructions():
    """Print Render deployment instructions."""
    print("\n" + "=" * 70)
    print("RENDER DEPLOYMENT (Free Tier Available)")
    print("=" * 70)
    print("\n1. Go to: https://render.com")
    print("2. Sign up/Login (you can use GitHub to sign in)")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect your GitHub account")
    print("5. Select your repository: video-analysis-api")
    print("6. Configure:")
    print("   - Name: visual-behavior-analysis-api")
    print("   - Environment: Python 3")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT")
    print("   - Plan: Free (or Starter)")
    print("7. Click 'Create Web Service'")
    print("\n8. After deployment:")
    print("   - Your API will be at: https://your-app.onrender.com")
    print("\n9. Generate API Key:")
    print("   - Use Render Shell: python api_keys.py")
    print("   - Or visit: https://your-app.onrender.com/api-key/generate")
    print("=" * 70)

def main():
    """Main deployment helper."""
    print("=" * 70)
    print("Visual Behavior Analysis API - Deployment Helper")
    print("=" * 70)
    
    # Prepare deployment
    if not prepare_deployment():
        print("\n[ERROR] Deployment preparation failed. Please fix the issues above.")
        return 1
    
    # Show GitHub instructions
    create_github_repo_instructions()
    
    input("\nPress Enter when you've created the GitHub repository...")
    
    # Get GitHub URL
    print("\n" + "=" * 70)
    print("STEP 2: Connect to GitHub")
    print("=" * 70)
    github_url = input("\nEnter your GitHub repository URL: ").strip()
    
    if not github_url:
        print("[ERROR] GitHub URL is required")
        return 1
    
    # Add remote and push
    print("\nAdding GitHub remote and preparing to push...")
    try:
        # Add all files
        subprocess.run(['git', 'add', '.'], check=True)
        print("[OK] Files staged")
        
        # Commit
        subprocess.run(['git', 'commit', '-m', 'Initial deployment - Visual Behavior Analysis API'], check=True)
        print("[OK] Files committed")
        
        # Add remote
        subprocess.run(['git', 'remote', 'add', 'origin', github_url], check=True)
        print(f"[OK] Remote added: {github_url}")
        
        # Push
        print("\nPushing to GitHub...")
        subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
        print("[SUCCESS] Code pushed to GitHub!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Git operation failed: {e}")
        print("\nYou can manually run:")
        print(f"  git remote add origin {github_url}")
        print("  git add .")
        print("  git commit -m 'Initial deployment'")
        print("  git push -u origin main")
        return 1
    
    # Show deployment options
    print("\n" + "=" * 70)
    print("STEP 3: Choose Deployment Platform")
    print("=" * 70)
    print("\n1. Railway (Recommended - Easiest)")
    print("2. Render (Free Tier Available)")
    print("3. Show both instructions")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        railway_deploy_instructions()
    elif choice == "2":
        render_deploy_instructions()
    else:
        railway_deploy_instructions()
        render_deploy_instructions()
    
    print("\n" + "=" * 70)
    print("DEPLOYMENT READY!")
    print("=" * 70)
    print("\nYour code is now on GitHub and ready to deploy!")
    print("Follow the instructions above to deploy on your chosen platform.")
    print("\nAfter deployment, don't forget to:")
    print("1. Generate your API key")
    print("2. Test the /health endpoint")
    print("3. Test the /analyze/frame endpoint")
    print("\nGood luck! 🚀")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDeployment cancelled by user.")
        sys.exit(1)

