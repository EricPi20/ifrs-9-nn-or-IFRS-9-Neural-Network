#!/bin/bash
# Helper script to push with GitHub Personal Access Token

echo "🚀 Git Push Helper"
echo "=================="
echo ""
echo "This script will help you push your commits to GitHub."
echo ""

# Check if token is provided as argument
if [ -z "$1" ]; then
    echo "Usage: ./push-with-token.sh YOUR_GITHUB_TOKEN"
    echo ""
    echo "Or run interactively:"
    echo "  git push origin main"
    echo "  (When prompted, enter your GitHub username and token as password)"
    echo ""
    read -p "Do you want to push interactively? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin main
    else
        echo "Please provide your token: ./push-with-token.sh YOUR_TOKEN"
        exit 1
    fi
else
    TOKEN=$1
    echo "Using provided token to push..."
    git remote set-url origin https://${TOKEN}@github.com/EricPi20/ifrs-9-nn-or-IFRS-9-Neural-Network.git
    git push origin main
    # Reset to original URL (without token)
    git remote set-url origin https://github.com/EricPi20/ifrs-9-nn-or-IFRS-9-Neural-Network.git
    echo ""
    echo "✅ Push complete!"
fi

