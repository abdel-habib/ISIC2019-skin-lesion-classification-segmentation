from pathlib import Path
from loguru import logger
import sys

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)

def log_to_file(text, filepath):
    '''A function that logs text to a file.
    
    Args:
        text ('str'): text to be logged.
        filepath ('Path'): path of the file

    Returns:
        None

    Example:
        >>> log_to_file("This is a log message.", Path("log.txt"))
        >>> logger.info("This is a loguru log message.")
    '''
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
        
        # Log the custom text
        with open(filepath, 'a') as file:
            file.write(text + '\n')

    except Exception as e:
        print(f"An error occurred: {e}")

