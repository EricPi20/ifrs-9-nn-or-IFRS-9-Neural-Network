# How to Push Your Commit to GitHub

## Step 1: Sign in to GitHub
The browser should be open to the GitHub login page. Sign in with your credentials.

## Step 2: Create Personal Access Token
After signing in, you'll be redirected to create a token. If not, go to:
https://github.com/settings/tokens/new

**On the token creation page:**
1. **Note**: Enter "Git Push Token" (or any name you like)
2. **Expiration**: Choose your preference (90 days, or No expiration)
3. **Select scopes**: Check the box for **`repo`** (this gives full repository access)
4. Click **"Generate token"** at the bottom
5. **IMPORTANT**: Copy the token immediately (it starts with `ghp_`). You won't see it again!

## Step 3: Push Your Commit

Once you have your token, come back here and I'll help you push!

Or run this command in your terminal:
```bash
cd "/Users/johnericpineda/Documents/Cursor/IFRS 9 NN"
git push origin main
```

When prompted:
- **Username**: `EricPi20`
- **Password**: Paste your token (not your GitHub password)

---

**Your commit ready to push:**
- Commit: `e019791` - "Increase file upload size limit from 100MB to 500MB"

