"""
Utility functions for loading and processing data.
ATMS 523 Module 6
"""

import os
import urllib.request
import pandas as pd


def load_data(url, local_filename, **read_csv_kwargs):
    """
    Load data from URL or local cache.
    Downloads raw file first if not cached, so you can inspect the original.
    
    Parameters:
    -----------
    url : str
        URL to download data from
    local_filename : str
        Local filename to save/load data
    **read_csv_kwargs : dict
        Additional arguments to pass to pd.read_csv (like sep, skiprows, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    if not os.path.exists(local_filename):
        print(f"Downloading raw file from: {url}")
        print(f"Saving to: {local_filename}")
        try:
            urllib.request.urlretrieve(url, local_filename)
            print(f"Data saved as: {local_filename}")
        except Exception as e:
            print(f"Error downloading: {e}")
            raise
    else:
        print(f"Using saved file: {local_filename}")
    
    # Read the file with the specified parameters
    data = pd.read_csv(local_filename, **read_csv_kwargs)
    return data

