"""
RIFT Neural Network Scorecard - Global Constants

This file contains constants used across the application,
particularly for input scaling and score calculation.
"""

# Input Scaling
# The upstream data transformation multiplies standardized log odds by -50
# We need to normalize this before feeding to the neural network
INPUT_SCALE_FACTOR = 50.0

# Score Range
SCORE_MIN = 0
SCORE_MAX = 100

# Default Model Settings
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.30

# Risk Level Thresholds
RISK_THRESHOLDS = {
    'EXCELLENT': 80,
    'GOOD': 60,
    'FAIR': 40,
    'POOR': 20,
    'VERY_POOR': 0
}

