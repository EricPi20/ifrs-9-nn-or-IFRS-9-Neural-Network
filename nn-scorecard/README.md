# Neural Network Scorecard Web Application

A professional web application for developing credit scorecards using neural networks. This tool trains neural network models on pre-processed credit risk data and converts them into interpretable scorecards with a 0-100 scale.

## Project Overview

- **Purpose**: Train neural network models for credit risk assessment
- **Input**: Pre-processed credit risk data with WoE (Weight of Evidence) transformed features (2-6 bins per feature)
- **Output**: Scorecard scaled 0-100 where 100 represents the lowest risk (best score)
- **Optimization**: Accuracy Ratio (AR) / Gini coefficient
- **Data Split**: Train/test split only (no validation set)

## Technical Stack

### Backend
- **Framework**: FastAPI with Python 3.11+
- **ML Framework**: PyTorch for neural networks
- **Data Processing**: pandas, scikit-learn, numpy

### Frontend
- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **Charts**: Recharts

## Project Structure

```
nn-scorecard/
├── backend/                 # FastAPI backend application
│   ├── app/
│   │   ├── main.py         # FastAPI application entry point
│   │   ├── config.py       # Configuration and settings
│   │   ├── models/         # Pydantic request/response models
│   │   ├── routers/        # API route handlers
│   │   ├── services/       # Business logic (training, scoring, etc.)
│   │   └── utils/          # Utility functions
│   ├── tests/              # Unit and integration tests
│   └── requirements.txt     # Python dependencies
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API service layer
│   │   └── types/          # TypeScript type definitions
│   ├── package.json        # Node.js dependencies
│   └── vite.config.ts      # Vite configuration
├── data/
│   ├── uploads/            # Uploaded CSV files
│   └── models/             # Saved model checkpoints
├── docker/                 # Docker configuration files
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`

### Docker Setup

1. Build and run with Docker Compose:
   ```bash
   cd docker
   docker-compose up --build
   ```

   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

## Usage

### 1. Upload Data

- Navigate to the Upload page
- Upload a CSV file containing pre-processed credit risk data
- Ensure the data has:
  - WoE transformed features (2-6 bins per feature)
  - A segment column (optional, for portfolio segmentation)
  - A target column (binary: 0/1)

### 2. Configure Training

- Go to the Training page
- Configure training parameters:
  - Test size (default: 0.2)
  - Batch size (default: 256)
  - Number of epochs (default: 100)
  - Learning rate (default: 0.001)
  - Hidden layer architecture
  - Dropout rate
  - Activation function

### 3. Train Model

- Start the training job
- Monitor progress in real-time
- The model optimizes for Accuracy Ratio (AR)

### 4. View Results

- Review model performance metrics:
  - Accuracy Ratio (AR) / Gini coefficient
  - AUC (Area Under ROC Curve)
  - KS (Kolmogorov-Smirnov) statistic
- View ROC curves and score distributions
- Generate scorecard (0-100 scale)

### 5. Calculate Scores

- Use the scoring API to calculate credit scores for new data
- Scores range from 0-100, where 100 represents the lowest risk

## API Endpoints

### Upload
- `POST /api/upload/` - Upload CSV file
- `GET /api/upload/{file_id}/info` - Get file metadata

### Training
- `POST /api/training/start` - Start training job
- `GET /api/training/{job_id}/status` - Get training status
- `GET /api/training/jobs` - List all training jobs

### Results
- `GET /api/results/{job_id}/metrics` - Get model evaluation metrics
- `POST /api/results/{job_id}/scorecard` - Generate scorecard
- `GET /api/results/{job_id}/scorecard` - Get existing scorecard

### Scoring
- `POST /api/scoring/calculate` - Calculate score for feature values
- `POST /api/scoring/batch` - Batch score calculation

## Development

### Running Tests

Backend tests:
```bash
cd backend
pytest
```

### Code Structure

- **Backend**: Follows FastAPI best practices with clear separation of concerns
- **Frontend**: Component-based architecture with TypeScript for type safety
- **Services**: Business logic separated from API routes
- **Models**: Pydantic schemas for request/response validation

## Configuration

Environment variables can be set in a `.env` file in the backend directory:

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
DEBUG=False
MAX_UPLOAD_SIZE=104857600
UPLOAD_DIR=data/uploads
MODEL_DIR=data/models
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

