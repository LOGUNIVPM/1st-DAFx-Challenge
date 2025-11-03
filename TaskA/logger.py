"""
Global logging module.
Provides a globally accessible print function that logs to both terminal and file.
"""

import os
import builtins
from datetime import datetime

# Global variables
_log_file_path = None
_original_print = builtins.print
_logging_initialized = False

def initialize_logging(experiment_folder):
    """Initialize the global logging system with the experiment folder."""
    global _log_file_path, _logging_initialized
    _log_file_path = os.path.join(experiment_folder, 'experiment.log')
    _logging_initialized = True
    
    # Override the built-in print function globally
    builtins.print = custom_print
    
def custom_print(*args, sep=' ', end='\n', file=None, flush=False):
    """Custom print function that writes to both terminal and log file."""
    # Always print to terminal using original print
    _original_print(*args, sep=sep, end=end, file=file, flush=flush)
    
    # Also write to log file if logging is initialized
    if _logging_initialized and _log_file_path and file is None:  # Only log if not redirecting to specific file
        message = sep.join(str(arg) for arg in args) + end
        try:
            with open(_log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(message)
                if flush:
                    log_file.flush()
        except Exception as e:
            # Fallback to original print if logging fails
            _original_print(f"[LOGGING ERROR]: {e}")

def get_log_file_path():
    """Get the current log file path."""
    return _log_file_path

def is_logging_initialized():
    """Check if logging has been initialized."""
    return _logging_initialized
