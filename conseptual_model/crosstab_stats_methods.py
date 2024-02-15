# load libraries
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import scipy as sp

def chi2(observed,threshold = .05):
    
    """
    Performs the Chi-Squared test of independence to evaluate if two categorical variables are related.

    Parameters:
    observed : array_like
        The contingency table containing the observed frequencies of cases.
    threshold : float, optional
        The significance level to determine if the variables can be considered correlated (default is .05).

    Returns:
    list
        A list containing the p-value and a boolean indicating whether the variables can be considered correlated.
    """
    chi2, p, dof, expected = chi2_contingency(observed)

    if p <= threshold:
        print('It can be correlated. p-value:{}'.format(p))
        chi2_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        chi2_result = False

    return [p, chi2_result]

def crosstab_chi2(arr1,arr2,threshold,**kwargs):

    """
    Creates a contingency table from two arrays and performs the Chi-Squared test of independence.

    Parameters:
    arr1 : array_like
        The first array or Series to be used in creating the contingency table.
    arr2 : array_like
        The second array or Series to be used in creating the contingency table.
    threshold : float
        The significance level for the Chi-Squared test.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to pd.crosstab().

    Returns:
    list
        A list containing the p-value and a boolean indicating whether the variables represented by arr1 and arr2 can be considered correlated.
    """
    
    arrays = {}

    iter = 1

    for arr in [arr1,arr2]:

        arr_copy = arr.copy()

        # arr_filled_value = arr_copy.value_counts().idxmax()
        # arr_new = arr_copy.fillna(arr_filled_value)

        # arrays[iter] = arr_new

        arrays[iter] = arr_copy

        iter += 1

    observed = pd.crosstab(arrays[1],arrays[2],**kwargs)
    observed_filled = observed.fillna(0)

    return chi2(observed_filled,threshold)


def chi2_residual(observed,print_ = False):
    
    """
    Performs the residual analysis after Chi-Squared test of independence to evaluate which levels are correlated.

    Parameters:
    observed : array_like
        The contingency table containing the observed frequencies of cases.

    Returns:
    list
        A dataframe containing the standardized residuals indicating whether the variables can be considered correlated.
        if the absolute value of the standardized residuals are over 1.96 (confidence interval 95%), it can be said that there is significant correlation.
    """
    chi2, p, dof, expected = chi2_contingency(observed)

    residuals = (observed - expected) / np.sqrt(expected)

    if print_ == True:
        print("standardized residuals :")
        print(residuals)

    return residuals