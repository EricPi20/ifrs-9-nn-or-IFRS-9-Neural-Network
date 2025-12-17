import React, { ReactNode } from 'react';

// Button
type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  loading?: boolean;
  icon?: ReactNode;
}

const buttonStyles: Record<ButtonVariant, string> = {
  primary: 'bg-[#1E3A5F] hover:bg-[#2D4A6F] text-white',
  secondary: 'bg-[#38B2AC] hover:bg-[#2C9A94] text-white',
  danger: 'bg-[#F56565] hover:bg-[#E53E3E] text-white',
  ghost: 'bg-transparent hover:bg-gray-100 text-gray-700'
};

export const Button: React.FC<ButtonProps> = ({ 
  variant = 'primary',
  loading = false,
  icon,
  children,
  disabled,
  className = '',
  ...props 
}) => {
  // Force navy blue for primary buttons
  const buttonStyle = variant === 'primary' 
    ? { backgroundColor: '#1E3A5F', color: '#FFFFFF', border: 'none' }
    : variant === 'secondary'
    ? { backgroundColor: '#38B2AC', color: '#FFFFFF', border: 'none' }
    : undefined;
  
  return (
  <button
    className={`
      inline-flex items-center justify-center gap-2 
      px-4 py-2 rounded-lg font-medium text-sm
      transition-colors duration-200
      disabled:opacity-50 disabled:cursor-not-allowed
      ${buttonStyles[variant]}
      ${className}
    `}
    style={buttonStyle}
    disabled={disabled || loading}
    {...props}
  >
    {loading ? (
      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    ) : icon}
    {children}
  </button>
  );
};

// Loading Spinner
interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
}

export const LoadingSpinner: React.FC<SpinnerProps> = ({ size = 'md' }) => {
  const sizes = { sm: 'h-4 w-4', md: 'h-8 w-8', lg: 'h-12 w-12' };
  return (
    <div className="flex justify-center items-center">
      <svg className={`animate-spin ${sizes[size]} text-[#38B2AC]`} fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    </div>
  );
};

// Alert
type AlertVariant = 'success' | 'warning' | 'error' | 'info';

interface AlertProps {
  variant: AlertVariant;
  title?: string;
  children: ReactNode;
  onClose?: () => void;
}

const alertStyles: Record<AlertVariant, { bg: string; border: string; icon: string }> = {
  success: { bg: 'bg-green-50', border: 'border-green-200', icon: '✓' },
  warning: { bg: 'bg-amber-50', border: 'border-amber-200', icon: '⚠' },
  error: { bg: 'bg-red-50', border: 'border-red-200', icon: '✕' },
  info: { bg: 'bg-blue-50', border: 'border-blue-200', icon: 'ℹ' }
};

export const Alert: React.FC<AlertProps> = ({ variant, title, children, onClose }) => {
  const styles = alertStyles[variant];
  return (
    <div className={`${styles.bg} ${styles.border} border rounded-lg p-4 flex gap-3`}>
      <span className="text-lg">{styles.icon}</span>
      <div className="flex-1">
        {title && <h4 className="font-semibold mb-1">{title}</h4>}
        <div className="text-sm">{children}</div>
      </div>
      {onClose && (
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
          ✕
        </button>
      )}
    </div>
  );
};

// Card
interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ title, children, className = '' }) => {
  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      )}
      <div>{children}</div>
    </div>
  );
};

// Badge
interface BadgeProps {
  children: ReactNode;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ children, className = '' }) => {
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${className}`}>
      {children}
    </span>
  );
};

// MetricCard
interface MetricCardProps {
  label: string;
  value: string | number;
  color?: 'navy' | 'teal';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({ 
  label, 
  value, 
  color = 'navy',
  className = '' 
}) => {
  const colorClasses = {
    navy: 'text-[#1E3A5F]',
    teal: 'text-[#38B2AC]'
  };

  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-4 ${className}`}>
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${colorClasses[color]}`}>
        {value}
      </div>
    </div>
  );
};

