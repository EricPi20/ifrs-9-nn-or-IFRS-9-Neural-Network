# Installation Guide

This guide will help you install and run the Neural Network Scorecard application on your desktop.

## Prerequisites

Before installing, make sure you have the following installed on your desktop:

1. **Python 3.11 or higher**
   - Check: `python3 --version` or `python --version`
   - Download: https://www.python.org/downloads/

2. **Node.js 18 or higher** (includes npm)
   - Check: `node --version` and `npm --version`
   - Download: https://nodejs.org/

3. **Git** (optional, if you need to clone the repository)
   - Check: `git --version`
   - Download: https://git-scm.com/downloads

## Installation Steps

### Option 1: Manual Installation (Recommended for Development)

#### Step 1: Navigate to the Project Directory

```bash
cd "/Users/EricPinedaWork/Documents/Cursor/IFRS 9 NN/nn-scorecard"
```

#### Step 2: Set Up the Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```

4. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Note:** This may take several minutes as it installs PyTorch and other ML libraries.

#### Step 3: Set Up the Frontend

1. Open a new terminal window and navigate to the frontend directory:
   ```bash
   cd "/Users/EricPinedaWork/Documents/Cursor/IFRS 9 NN/nn-scorecard/frontend"
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

   **Note:** This may take a few minutes to download all packages.

#### Step 4: Run the Application

You need to run both the backend and frontend servers:

**Terminal 1 - Backend:**
```bash
cd "/Users/EricPinedaWork/Documents/Cursor/IFRS 9 NN/nn-scorecard/backend"
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd "/Users/EricPinedaWork/Documents/Cursor/IFRS 9 NN/nn-scorecard/frontend"
npm run dev
```

#### Step 5: Access the Application

- **Frontend:** Open your browser and go to `http://localhost:5173`
- **Backend API:** Available at `http://localhost:8000`
- **API Documentation:** `http://localhost:8000/docs` (Swagger UI)

---

### Option 2: Docker Installation (Recommended for Production)

If you have Docker and Docker Compose installed:

1. Navigate to the docker directory:
   ```bash
   cd "/Users/EricPinedaWork/Documents/Cursor/IFRS 9 NN/nn-scorecard/docker"
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

3. Access the application:
   - **Frontend:** `http://localhost:3000`
   - **Backend API:** `http://localhost:8000`

To stop the containers:
```bash
docker-compose down
```

---

## Troubleshooting

### Backend Issues

**Problem:** `python3: command not found`
- **Solution:** Install Python 3.11+ from https://www.python.org/downloads/

**Problem:** `pip: command not found`
- **Solution:** Make sure Python is installed correctly. Try `python -m pip` instead of `pip`

**Problem:** PyTorch installation fails
- **Solution:** This is normal on some systems. Try installing PyTorch separately:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

**Problem:** Port 8000 already in use
- **Solution:** Change the port in the uvicorn command:
  ```bash
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
  ```

### Frontend Issues

**Problem:** `node: command not found`
- **Solution:** Install Node.js 18+ from https://nodejs.org/

**Problem:** `npm: command not found`
- **Solution:** Node.js should include npm. Reinstall Node.js if needed.

**Problem:** Port 5173 already in use
- **Solution:** Vite will automatically use the next available port, or you can specify one:
  ```bash
  npm run dev -- --port 3001
  ```

**Problem:** `npm install` fails
- **Solution:** 
  - Delete `node_modules` folder and `package-lock.json`
  - Run `npm install` again
  - If issues persist, try `npm cache clean --force`

### General Issues

**Problem:** Can't connect frontend to backend
- **Solution:** 
  - Make sure both servers are running
  - Check that the backend is running on port 8000
  - Check browser console for CORS errors
  - Verify the API URL in `frontend/src/services/api.ts`

**Problem:** Virtual environment not activating
- **Solution:** 
  - Make sure you're in the correct directory
  - On Windows, use `venv\Scripts\activate.bat` or PowerShell: `venv\Scripts\Activate.ps1`
  - You may need to allow script execution in PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Quick Start Scripts

### macOS/Linux

Create a file `start-backend.sh`:
```bash
#!/bin/bash
cd "$(dirname "$0")/nn-scorecard/backend"
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Create a file `start-frontend.sh`:
```bash
#!/bin/bash
cd "$(dirname "$0")/nn-scorecard/frontend"
npm run dev
```

Make them executable:
```bash
chmod +x start-backend.sh start-frontend.sh
```

### Windows

Create a file `start-backend.bat`:
```batch
@echo off
cd "%~dp0nn-scorecard\backend"
call venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
```

Create a file `start-frontend.bat`:
```batch
@echo off
cd "%~dp0nn-scorecard\frontend"
npm run dev
pause
```

---

## Next Steps

Once the application is running:

1. **Upload Data:** Go to the Upload page and upload a CSV file with your credit risk data
2. **Configure Training:** Set up your training parameters on the Training page
3. **Train Model:** Start training and monitor progress
4. **View Results:** Check model metrics and generate scorecards

For more details, see the main [README.md](nn-scorecard/README.md) file.

---

## Need Help?

If you encounter any issues not covered here:
1. Check the main README.md for more information
2. Review the error messages in the terminal
3. Check browser console for frontend errors
4. Verify all prerequisites are installed correctly

