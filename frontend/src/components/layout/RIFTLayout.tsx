import React, { ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { RIFTLogo } from '../common/RIFTLogo';
import {
  HomeIcon,
  CloudArrowUpIcon,
  CpuChipIcon,
  ChartBarIcon,
  FolderIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';

interface LayoutProps {
  children: ReactNode;
}

const navItems = [
  { path: '/', label: 'Dashboard', icon: HomeIcon },
  { path: '/upload', label: 'Upload', icon: CloudArrowUpIcon },
  { path: '/train', label: 'Train', icon: CpuChipIcon },
  { path: '/results', label: 'Results', icon: ChartBarIcon },
  { path: '/models', label: 'Models', icon: FolderIcon },
  { path: '/settings', label: 'Settings', icon: Cog6ToothIcon },
];

export const RIFTLayout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-[#F7FAFC]">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <RIFTLogo size="md" variant="full" />
            
            {/* Navigation */}
            <nav className="hidden md:flex space-x-1">
              {navItems.map(({ path, label, icon: Icon }) => (
                <NavLink
                  key={path}
                  to={path}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-[#1E3A5F] text-white'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <Icon className="w-5 h-5" />
                  {label}
                </NavLink>
              ))}
            </nav>

            {/* User Menu */}
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">Globe Telecom</span>
              <div className="w-8 h-8 bg-[#38B2AC] rounded-full flex items-center justify-center text-white font-semibold">
                GT
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center text-sm text-gray-500">
            <span>RIFT NN Scorecard v4.0</span>
            <span>© 2024 Globe Telecom - Finance AI Department</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default RIFTLayout;

