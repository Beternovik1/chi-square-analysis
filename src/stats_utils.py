import pandas as pd
from scipy.stats import chi2_contingency, chisquare
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
    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    return observed_table, expected_table, chi2, p_value, cramer_v

def analyze_goodness_of_fit(observed_freqs, expected_freqs):
    """
    Computes the Chi-square Goodness of Fit test.
    Target: Exercises 5, 6, 10.
    
    Parameters:
    - observed_freqs: list or array of observed empirical frequencies.
    - expected_freqs: list or array of theoretical expected frequencies.
    """
    # Ensure inputs are numpy arrays for element-wise operations if needed later
    obs = np.array(observed_freqs)
    exp = np.array(expected_freqs)
    
    # Chi-square goodness of fit test
    chi2, p_value = chisquare(f_obs=obs, f_exp=exp)
    
    return chi2, p_value
    