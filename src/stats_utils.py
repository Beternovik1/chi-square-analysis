import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np


def analyze_independence(df, col1, col2):
    """
    computes the chi-square test of independence for two 
    categorical columns
    """
    # Contigency table (observed frequencies)
    observed_table = pd.crosstab(df[col1], df[col2])

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(observed_table)

    # format expected frequencies into a df 
    expected_table = pd.DataFrame(
        expected,
        index=observed_table.index,
        columns=observed_table.columns
    ).round(2)

    # calculate total number of observations in contigency table
    n = observed_table.sum().sum()

    
    # find min dimension - 1 or we can either use rows-1 or cols-1
    min_dim = min(observed_table.shape) - 1

    # Appying cramer's v formula
    cramer_v = np.sqrt(chi2/(n*min_dim))

    return observed_table, expected_table, chi2, p_value, cramer_v

# if __name__ == "__main__":

    