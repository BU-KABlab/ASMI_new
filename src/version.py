"""
Version information for ASMI - Automated Soft Matter Indenter

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

__version__ = "2.0.0"
__version_info__ = (2, 0, 0)
__author__ = "Hongrui Zhang"
__email__ = "hz622@bu.edu"  # Add your email if desired
__license__ = "MIT"
__copyright__ = f"Copyright 2025 {__author__}"

def get_version():
    """Return the version string."""
    return __version__

def get_version_info():
    """Return the version tuple."""
    return __version_info__

def get_full_version():
    """Return full version information."""
    return f"ASMI v{__version__} by {__author__} ({__license__} License)"
