# How to Push to GitHub

## Step 1: Create Repository on GitHub
1. Go to https://github.com and sign in
2. Click the "+" icon → "New repository"
3. Name it (e.g., `ifrs-9-nn`)
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

## Step 2: Push Your Code

After creating the repository, run these commands (replace `YOUR_USERNAME` and `REPO_NAME`):

```bash
# Add all files
git add .

# Make initial commit
git commit -m "Initial commit"

# Add remote (replace with your actual repository URL)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## If you need to authenticate:
- For HTTPS: You'll need a Personal Access Token (Settings → Developer settings → Personal access tokens)
- For SSH: Make sure your SSH key is added to your GitHub account

