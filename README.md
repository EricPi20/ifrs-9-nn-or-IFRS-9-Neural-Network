# IFRS 9 Neural Network Scorecard

A web application for developing credit scorecards using neural networks.

## Prerequisites

- Python 3.11+
- Node.js 18+

## Installation

### Backend Setup

On their computer:

```bash
# Clone the repository
git clone https://github.com/EricPi20/ifrs-9-nn-or-IFRS-9-Neural-Network.git
cd ifrs-9-nn-or-IFRS-9-Neural-Network/nn-scorecard/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then the backend API will be available at `http://localhost:8000`

### Frontend Setup

In a new terminal:

```bash
# Navigate to frontend directory
cd ifrs-9-nn-or-IFRS-9-Neural-Network/nn-scorecard/frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Then open `http://localhost:5173` in their browser.

## Troubleshooting

### "Failed to fetch" error when uploading files

If you see a "Failed to fetch" error when trying to upload a file:

1. **Make sure the backend is running:**
   - Check that you see the backend startup message in the terminal
   - Visit `http://localhost:8000/health` in your browser - it should return `{"status": "healthy"}`

2. **Check the frontend API URL:**
   - The frontend defaults to `http://localhost:8000`
   - If your backend is on a different port, create a `.env` file in `nn-scorecard/frontend/`:
     ```
     VITE_API_URL=http://localhost:8000
     ```
   - Then restart the frontend server

3. **Verify CORS settings:**
   - The backend allows requests from `http://localhost:5173` by default
   - If your frontend runs on a different port, update the backend `.env` file in `nn-scorecard/backend/`:
     ```
     CORS_ORIGINS=http://localhost:5173,http://localhost:3000
     ```
   - Then restart the backend server

4. **Check browser console:**
   - Open browser DevTools (F12) and check the Console tab for detailed error messages

## Usage

1. Upload your CSV file with pre-processed credit risk data
2. Configure training parameters
3. Train the neural network model
4. View results and generate scorecards

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

