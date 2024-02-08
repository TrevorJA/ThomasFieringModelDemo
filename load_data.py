# Load example data
import pandas as pd
def load_example_data():
    """
    Loads example data from the package.

    Returns:
        pd.DataFrame: Example data.
    """
    data = pd.read_csv(f'./usgs_monthly_streamflow_cms.csv', 
                       index_col=0, parse_dates=True)
    return data