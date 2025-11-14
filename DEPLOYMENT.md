# Vercel Deployment - Student Placement Predictor

## Quick Deploy to Vercel

### Option 1: Deploy via Vercel CLI (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Navigate to project directory**:
   ```bash
   cd /home/gabbru/Desktop/harshit
   ```

3. **Login to Vercel**:
   ```bash
   vercel login
   ```

4. **Deploy**:
   ```bash
   vercel
   ```
   - Follow the prompts
   - Answer "yes" to setup and deploy
   - Choose project name: `student-placement-predictor`
   - Select scope/team
   - Deploy!

5. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

### Option 2: Deploy via Vercel Dashboard (Easiest)

1. **Push to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Student placement predictor"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to Vercel**:
   - Visit https://vercel.com
   - Sign up/Login with GitHub
   - Click "New Project"
   - Import your GitHub repository
   - Configure:
     - Framework Preset: **Other**
     - Root Directory: `./` (leave as default)
     - Build Command: (leave empty - no build needed)
     - Output Directory: `./` (leave as default)
   - Click "Deploy"

### Option 3: Drag & Drop (Simplest)

1. **Create a deployment folder**:
   ```bash
   mkdir deploy
   cp index.html models.json deploy/
   ```

2. **Go to Vercel**:
   - Visit https://vercel.com
   - Sign up/Login
   - Drag and drop the `deploy` folder
   - Done!

## Files Needed for Deployment

Your project already has all necessary files:
- ✅ `index.html` - Main application
- ✅ `models.json` - Model coefficients
- ✅ `vercel.json` - Vercel configuration

## Optional: Environment Setup

If you want to exclude certain files from deployment, create a `.vercelignore` file:

```
venv/
*.py
*.png
Placement.csv
README.md
.git/
__pycache__/
```

## Post-Deployment

After deployment, Vercel will give you:
- **Preview URL**: `https://your-project-name-xxxxx.vercel.app`
- **Production URL**: `https://your-project-name.vercel.app`

You can also configure a custom domain in Vercel settings.

## Troubleshooting

### Issue: models.json not found
- Ensure `models.json` is in the root directory
- Check browser console for 404 errors
- Verify the file is included in deployment

### Issue: CORS errors
- Not applicable for this static site (everything runs client-side)

### Issue: Chart.js not loading
- Check internet connection (Chart.js loads from CDN)
- Verify CDN URL in index.html

## Commands Summary

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Deploy to production
vercel --prod

# Check deployment status
vercel ls

# View logs
vercel logs
```

## Your Live URL

After deployment, your app will be available at:
`https://student-placement-predictor.vercel.app` (or similar)

Share this URL with anyone to use your placement predictor!
