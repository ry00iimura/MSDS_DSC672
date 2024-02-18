# load libraries
from scipy.stats import chi2_contingency
import scikit_posthocs as skp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import pandas as pd
import scipy as sp
import b_stats_approach as b
import utility as u

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


def tukey(quantified_val,qualified_val):
    
    """
    Performs Tukey's Honest Significant Difference (HSD) test to compare means between multiple groups.
    
    Parameters:
    quantified_val : pandas.Series
        The numeric values corresponding to different groups.
    qualified_val : pandas.Series
        The group labels for the numeric values.
    qualified_val_filled : int or float
        The value used to fill missing entries in the qualified_val series.
    quantified_val_filled : int or float
        The value used to fill missing entries in the quantified_val series.
    
    Returns:
    None
        This function prints the result of the Tukey HSD test but does not return any value.
    """

    arr_list = u.array_split(qualified_val,quantified_val)
    data_arr = np.hstack(arr_list)

    elemet_counts = [len(arrays) for arrays in arr_list]
    ind_arr = np.repeat(qualified_val.unique(),elemet_counts)

    print(pairwise_tukeyhsd(data_arr,ind_arr))


def steel_dwass(quantified_val,qualified_val):

    """
    Performs the Steel-Dwass-Critchlow-Fligner multiple comparison test for nonparametric data.
    
    Parameters:
    quantified_val : pandas.Series
        The numeric values corresponding to different groups.
    qualified_val : pandas.Series
        The group labels for the numeric values.
    qualified_val_filled : int or float
        The value used to fill missing entries in the qualified_val series.
    quantified_val_filled : int or float
        The value used to fill missing entries in the quantified_val series.
    
    Returns:
    DataFrame
        A DataFrame containing the p-values of the pairwise comparisons from the Steel-Dwass test.
    """

    arr_list = u.array_split(qualified_val,quantified_val)
    data_arr = np.hstack(arr_list)

    element_counts = [len(arrays) for arrays in arr_list]
    ind_arr = np.repeat(qualified_val.unique(),element_counts)

    data = pd.DataFrame(data_arr, index = ind_arr,columns = ['values']).reset_index()

    return skp.posthoc_dscf(data, val_col='values', group_col='index')