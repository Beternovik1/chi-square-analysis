import pandas as pd
from scipy.stats import chi2_contingency



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

    return observed_table, expected_table, chi2, p_value

# if __name__ == "__main__":

    