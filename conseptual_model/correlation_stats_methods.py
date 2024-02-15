# load libraries
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import pandas as pd
import scipy as sp
import stats_methods as sm
import scikit_posthocs as skp

def pearson(arr1,arr2):
    """
    Calculates the Pearson correlation coefficient between two arrays.

    Parameters:
    arr1 : array_like
        The first set of observations.
    arr2 : array_like
        The second set of observations.

    Returns:
    tuple
        The Pearson correlation coefficient, the two-tailed p-value, and the name of the test ('Pearson').
    """
    corr , pvalue = sp.stats.pearsonr(arr1,arr2)
    print('Pearson"s correlation :{0} p-value:{1}'.format(round(corr,3),round(pvalue,3)))
    return corr, pvalue, 'Peason'

def spearman(arr1,arr2):
    """
    Calculates the Spearman rank-order correlation coefficient between two arrays.

    Parameters:
    arr1 : array_like
        The first set of observations.
    arr2 : array_like
        The second set of observations.

    Returns:
    tuple
        The Spearman correlation coefficient, the two-tailed p-value, and the name of the test ('Spearman').
    """
    corr , pvalue = sp.stats.spearmanr(arr1,arr2)
    print('Spearman"s correlation:{0} p-value:{1}'.format(round(corr,3),round(pvalue,3)))
    return corr, pvalue , 'Spearman'

def cramers_v(x, y):
    '''
    Calc Cramer's V.

    Parameters
    ----------
    x : {numpy.ndarray, pandas.Series}
    y : {numpy.ndarray, pandas.Series}

    returns corr, p-value(nan), 'Cramers V'
    arr1 must be categorical variable
    arr2 must be categorical variable

    correlation ratio is often denoted with rc

    if rc >= .5 then significantly strong correlation
    if .25 <= rc < .5 then somewhat strong correlation
    if .1 <= rc < .25 then weak correlation
    else not correlated
    '''

    # cross tabluation
    table = np.array(pd.crosstab(x, y))
    
    # measured variable
    n = table.sum()
    
    # column-wise total
    colsum = table.sum(axis=0)
    
    # row-wise total
    rowsum = table.sum(axis=1)
    
    # expectation
    expect = np.outer(rowsum, colsum) / n
    
    # chi2
    chisq = np.sum((table - expect) ** 2 /expect)

    # corr(cramer'V)
    corr = np.sqrt(chisq / (n * (min(table.shape) -1)))
    
    return corr, np.NaN, 'Cramers V'

def corr_ratio(arr1,arr2):
    '''
    returns corr, p-value(nan), 'correlation ratio'
    arr1 must be categorical variables
    arr2 must be numeric variables

    correlation ratio is often denoted with mu_square(m2)

    if m2 >= .5 then significantly strong correlation
    if .25 <= m2 < .5 then somewhat correlation
    if .1 <= m2 < .25 then weak correlation
    else not correlated
    '''
    # compute total variance
    all_var = ((arr2 - arr2.mean()) ** 2).sum()

    # compute intraclass variance
    intra_class_var = sum([((arr2[arr1 == i] - arr2[ arr1== i].mean()) ** 2).sum() for i in np.unique(arr1)])

    # compute interclass variance
    inter_class_var = all_var - intra_class_var

    # compute correlation ratio
    corr = inter_class_var / all_var
    print('compute correlation ratio:{}'.format(round(corr,3)))
    return corr, np.NaN, 'correlation ratio'


def tukey(quantified_val,qualified_val,qualified_val_filled,quantified_val_filled):
    
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

    quantified_val = quantified_val.fillna(quantified_val_filled)
    qualified_val = qualified_val.fillna(qualified_val_filled)

    arr_list = sm.array_split(qualified_val,quantified_val,qualified_val_filled,quantified_val_filled)
    data_arr = np.hstack(arr_list)

    elemet_counts = [len(arrays) for arrays in arr_list]
    ind_arr = np.repeat(qualified_val.unique(),elemet_counts)

    print(pairwise_tukeyhsd(data_arr,ind_arr))


def steel_dwass(quantified_val,qualified_val,qualified_val_filled,quantified_val_filled):

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

    quantified_val = quantified_val.fillna(quantified_val_filled)
    qualified_val = qualified_val.fillna(qualified_val_filled)

    arr_list = sm.array_split(qualified_val,quantified_val,qualified_val_filled,quantified_val_filled)
    data_arr = np.hstack(arr_list)

    elemet_counts = [len(arrays) for arrays in arr_list]
    ind_arr = np.repeat(qualified_val.unique(),elemet_counts)

    data = pd.DataFrame(data_arr, index = ind_arr,columns = ['values']).reset_index()

    return skp.posthoc_dscf(data, val_col='values', group_col='index')