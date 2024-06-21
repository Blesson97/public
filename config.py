# config.py

"""
This module provides configuration settings for the application.
"""

# Default model name
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"

def set_default_model_name(model_name):
    """
    Sets the default model name for the application.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    global DEFAULT_MODEL_NAME
    DEFAULT_MODEL_NAME = model_name


# Constants for text colors
COLOR_WHITE = "\033[37m"
COLOR_GREEN = "\033[32m"
COLOR_RESET = "\033[0m"