# load libraries
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import scipy as sp

def chi2(observed,print_,threshold = .05):
    
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

        chi2_result = True

        if print_ == True:
            print('-----------------')
            print('Chi2-test : it can be correlated. p-value:{}'.format(p))
            print('arr1 : {}'.format(observed.index.name))
            print('arr2 : {}'.format(observed.columns.name))
            print('-----------------')

    else:

        chi2_result = False

        if print_ == True:
            print('-----------------')
            print('Chi2-test : the null hypothesis cannot be rejected.')
            print('arr1 : {}'.format(observed.index.name))
            print('arr2 : {}'.format(observed.columns.name))
            print('-----------------')

    return [p, chi2_result]

def crosstab_chi2(arr1,arr2,print_,threshold,**kwargs):

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
    
    observed = pd.crosstab(arr1,arr2,**kwargs)
    observed_filled = observed.fillna(0)

    return chi2(observed_filled,print_,threshold)


def one_way_ANOVA(print_,threshold,*observed):

    """
    Performs a one-way ANOVA test to compare the means of two or more groups.

    Parameters:
    threshold : float
        The significance level for the test.
    *observed : multiple array_like
        Variable length argument list of arrays representing the groups to compare.

    Returns:
    list
        A list containing the p-value and a boolean indicating whether at least two groups have different means.
    """

    s , p = sp.stats.f_oneway(*observed)

    if p <= threshold:

        ANOVA_result = True

        if print_ == True:
            print('-----------------')
            print('one-way ANOVA test : the can be said that two or more groups does not have the same population mean. p-value:{}'.format(p))
            print('-----------------')
        
    else:

        ANOVA_result = False

        if print_ == True:
            print('-----------------')
            print('one-way ANOVA test : the null hypothesis cannot be rejected.')
            print('-----------------')

    return [p, ANOVA_result]    

def kruskal(print_,threshold,*observed):

    """
    Performs Kruskal-Wallis H-test for comparing the medians of two or more groups.

    Parameters:
    threshold : float
        The significance level for the test.
    *observed : multiple array_like
        Variable length argument list of arrays representing the groups to compare.

    Returns:
    list
        A list containing the p-value and a boolean indicating whether the population medians of all of the groups are not equal.
    """

    s , p = sp.stats.kruskal(*observed)

    if p <= threshold:
        
        kruskal_result = True

        if print_ == True:
            print('-----------------')
            print('Kruskal-Wallis H-test: The can be said that the population median of all of the groups are not equal. p-value:{}'.format(p))
            print('-----------------')
        
    else:

        kruskal_result = False

        if print_ == True:
            print('-----------------')
            print('Kruskal-Wallis H-test: The null hypothesis cannot be rejected.')
            print('-----------------')
        
    return [p, kruskal_result]


def t_test(arr1,arr2,test_type,print_,threshold = .05):

    """
    Performs different types of t-tests (Student's t-test, Welch's t-test, or paired t-test) on two arrays.

    Parameters:
    arr1 : array_like
        The first sample data array.
    arr2 : array_like
        The second sample data array.
    test_type : str
        Specifies the type of t-test to perform. Acceptable values are 'student_t', 'welch_t', or 'related'.
    threshold : float, optional
        The significance level to determine if the samples can be considered significantly different (default is .05).

    Returns:
    list
        A list containing the p-value and a boolean indicating whether the samples can be considered significantly different according to the specified t-test.
        
    Notes:
    - 'student_t': Assumes equal variance between the two samples.
    - 'welch_t': Does not assume equal variance between the two samples, suitable for samples with unequal variances and/or unequal sample sizes.
    - 'related': Assumes that the two samples are related or paired, suitable for repeated measurements on the same subjects.
    """

    if test_type == 'student_t':
        statistic, p = sp.stats.ttest_ind(arr1,arr2)

    elif test_type == 'welch_t':
        statistic, p = sp.stats.ttest_ind(arr1,arr2,equal_var = False)
    
    elif test_type == 'related':
        statistic, p = sp.stats.ttest_rel(arr1,arr2)

    if p <= threshold:

        ttest_result = True
        
        if print_ == True:
            print('-----------------')
            print('T-test :It can be correlated. p-value:{}'.format(p))
            print('arr1 : {}'.format(arr1.index.name))
            print('arr2 : {}'.format(arr2.name))
            print('-----------------')

    else:

        ttest_result = False

        if print_ == True:
            print('-----------------')
            print('T-test : The null hypothesis cannot be rejected.')
            print('arr1 : {}'.format(arr1.index.name))
            print('arr2 : {}'.format(arr2.name))
            print('-----------------')
    
    return [p, ttest_result] 