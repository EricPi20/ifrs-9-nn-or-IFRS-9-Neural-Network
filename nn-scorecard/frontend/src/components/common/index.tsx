import React, { ReactNode } from 'react';

// Card
interface CardProps {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
}

export const Card: React.FC<CardProps> = ({ 
  title, 
  subtitle, 
  children, 
  className = '',
  actions 
}) => (
  <div className={`bg-white rounded-xl shadow-md overflow-hidden ${className}`} style={{ backgroundColor: '#FFFFFF' }}>
    {(title || actions) && (
      <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center">
        <div>
          {title && <h3 className="text-lg font-semibold text-[#1E3A5F]">{title}</h3>}
          {subtitle && <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>}
        </div>
        {actions && <div>{actions}</div>}
      </div>
    )}
    <div className="p-6">{children}</div>
  </div>
);

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

// Badge
type BadgeVariant = 'success' | 'warning' | 'error' | 'info' | 'neutral';

interface BadgeProps {
  variant?: BadgeVariant;
  children: ReactNode;
}

const badgeStyles: Record<BadgeVariant, string> = {
  success: 'bg-green-100 text-green-800',
  warning: 'bg-amber-100 text-amber-800',
  error: 'bg-red-100 text-red-800',
  info: 'bg-blue-100 text-blue-800',
  neutral: 'bg-gray-100 text-gray-800'
};

export const Badge: React.FC<BadgeProps> = ({ variant = 'neutral', children }) => (
  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badgeStyles[variant]}`}>
    {children}
  </span>
);

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

// Metric Card
interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'navy' | 'teal' | 'green' | 'amber' | 'red';
}

const metricColors = {
  navy: 'from-[#1E3A5F] to-[#2D4A6F]',
  teal: 'from-[#38B2AC] to-[#4FD1C5]',
  green: 'from-[#48BB78] to-[#68D391]',
  amber: 'from-[#ED8936] to-[#F6AD55]',
  red: 'from-[#F56565] to-[#FC8181]'
};

export const MetricCard: React.FC<MetricCardProps> = ({ 
  label, 
  value, 
  subtitle,
  color = 'navy' 
}) => (
  <div className={`bg-gradient-to-br ${metricColors[color]} rounded-xl p-6 text-white`}>
    <div className="text-sm opacity-90 uppercase tracking-wide">{label}</div>
    <div className="text-3xl font-bold font-mono mt-2">{value}</div>
    {subtitle && <div className="text-sm opacity-75 mt-1">{subtitle}</div>}
  </div>
);

// Tooltip
type TooltipPosition = 'top' | 'bottom' | 'left' | 'right';

interface TooltipProps {
  content: ReactNode;
  position?: TooltipPosition;
  children: ReactNode;
  className?: string;
}

const tooltipPositions: Record<TooltipPosition, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2'
};

const tooltipArrows: Record<TooltipPosition, string> = {
  top: 'top-full left-1/2 -translate-x-1/2 border-t-gray-900',
  bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-gray-900',
  left: 'left-full top-1/2 -translate-y-1/2 border-l-gray-900',
  right: 'right-full top-1/2 -translate-y-1/2 border-r-gray-900'
};

export const Tooltip: React.FC<TooltipProps> = ({ 
  content, 
  position = 'top',
  children,
  className = ''
}) => {
  const [isVisible, setIsVisible] = React.useState(false);
  
  return (
    <div 
      className={`relative inline-block ${className}`}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div className={`absolute z-50 ${tooltipPositions[position]}`}>
          <div className="bg-gray-900 text-white text-sm rounded-lg px-3 py-2 shadow-lg whitespace-nowrap">
            {content}
          </div>
          <div className={`absolute ${tooltipArrows[position]} border-4 border-transparent`} />
        </div>
      )}
    </div>
  );
};

