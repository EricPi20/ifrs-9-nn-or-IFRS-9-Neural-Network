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

## Usage

1. Upload your CSV file with pre-processed credit risk data
2. Configure training parameters
3. Train the neural network model
4. View results and generate scorecards

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

