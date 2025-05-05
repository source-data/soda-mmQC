"""SODA MMQC package."""

import logging
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
package_dir = Path(__file__).parent
logs_dir = package_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# Create logger instance
logger = logging.getLogger(__name__)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)  # Console shows INFO and above

# Create file handler
log_file = logs_dir / "soda_mmqc.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # File shows DEBUG and above

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Set the logger level to the lowest level we want to capture
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to prevent duplicate logging
logger.propagate = False

