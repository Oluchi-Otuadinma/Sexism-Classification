import os
import re
import pandas as pd

def load_raw_csvs(raw_path):
    """
    Loads all CSV files from a raw dataset directory.
    Returns:
        dfs (dict): {cleaned_name: pandas.DataFrame}
    """

    csv_files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]
    dfs = {}

    for file in csv_files:
        var_name = os.path.splitext(file)[0]

        # clean the variable name
        var_name = var_name.lower()
        var_name = re.sub(r"\s+|-", "_", var_name)
        var_name = re.sub(r"\(\d+\)", "", var_name)
        var_name = re.sub(r"_+", "_", var_name)
        var_name = var_name.strip("_")

        df = pd.read_csv(os.path.join(raw_path, file))
        dfs[var_name] = df

    return dfs
