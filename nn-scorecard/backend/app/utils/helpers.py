"""
Utility Helper Functions

This module contains general-purpose utility functions used throughout
the application.
"""

from pathlib import Path
from typing import Optional


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Validate that a file has an allowed extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.txt'])
        
    Returns:
        True if extension is allowed, False otherwise
    """
    return any(filename.lower().endswith(ext.lower()) for ext in allowed_extensions)


def format_number(value: float, decimals: int = 4) -> str:
    """
    Format a number to a specified number of decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}"

