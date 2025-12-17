import React from 'react';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'full' | 'icon';
  className?: string;
}

export const RIFTLogo: React.FC<LogoProps> = ({ 
  size = 'md', 
  variant = 'full',
  className = '' 
}) => {
  const sizes = {
    sm: { icon: 24, text: 'text-lg' },
    md: { icon: 32, text: 'text-2xl' },
    lg: { icon: 48, text: 'text-4xl' }
  };

  const { icon, text } = sizes[size];

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Diamond Icon */}
      <svg 
        width={icon} 
        height={icon} 
        viewBox="0 0 48 48" 
        fill="none"
      >
        {/* Outer diamond */}
        <path
          d="M24 4L44 24L24 44L4 24L24 4Z"
          fill="#1E3A5F"
        />
        {/* Inner accent */}
        <path
          d="M24 12L36 24L24 36L12 24L24 12Z"
          fill="#38B2AC"
        />
        {/* Center */}
        <circle cx="24" cy="24" r="4" fill="white" />
      </svg>

      {variant === 'full' && (
        <div className="flex flex-col">
          <span className={`font-bold ${text} text-[#1E3A5F] tracking-tight`}>
            RIFT
          </span>
          <span className="text-xs text-[#718096] -mt-1">
            Risk Intelligence & Finance Technology
          </span>
        </div>
      )}
    </div>
  );
};

export default RIFTLogo;

