/**
 * Main App Component
 * 
 * Root component with routing setup.
 */

import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { Training } from './pages/Training';
import Results from './pages/Results';
import ScorecardView from './pages/ScorecardView';
import ConfigView from './pages/ConfigView';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50" style={{ backgroundColor: '#F9FAFB', minHeight: '100vh' }}>
        <nav className="bg-white shadow-sm">
          <div className="container mx-auto px-4">
            <div className="flex justify-between items-center h-16">
              <div className="flex space-x-8">
                <Link to="/" className="text-xl font-bold text-blue-600">
                  NN Scorecard
                </Link>
                <div className="flex space-x-4">
                  <Link
                    to="/"
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Dashboard
                  </Link>
                  <Link
                    to="/upload"
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Upload
                  </Link>
                  <Link
                    to="/training"
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Training
                  </Link>
                  <Link
                    to="/results"
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Results
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/training" element={<Training />} />
          <Route path="/results" element={<Results />} />
          <Route path="/results/:jobId/scorecard" element={<ScorecardView />} />
          <Route path="/results/:jobId/config" element={<ConfigView />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

