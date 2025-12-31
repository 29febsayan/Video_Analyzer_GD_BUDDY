@echo off
REM Setup script for production deployment (Windows)

echo ==========================================
echo Visual Behavior Analysis API - Deployment Setup
echo ==========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    git branch -M main
    echo [OK] Git repository initialized
) else (
    echo [OK] Git repository already exists
)

REM Check Dockerfile
if not exist "Dockerfile" (
    echo [ERROR] Dockerfile not found!
    exit /b 1
) else (
    echo [OK] Dockerfile found
)

REM Check requirements.txt
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    exit /b 1
) else (
    echo [OK] requirements.txt found
)

REM Check main.py
if not exist "main.py" (
    echo [ERROR] main.py not found!
    exit /b 1
) else (
    echo [OK] main.py found
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Create a GitHub repository
echo 2. Add remote: git remote add origin YOUR_REPO_URL
echo 3. Commit and push:
echo    git add .
echo    git commit -m "Initial deployment setup"
echo    git push -u origin main
echo 4. Deploy on Railway or Render (see QUICK_DEPLOY.md)
echo.

pause

