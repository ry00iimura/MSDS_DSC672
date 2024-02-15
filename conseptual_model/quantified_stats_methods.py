# load libraries
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import scipy as sp

def t_test(arr1,arr2,test_type,threshold = .05):

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
        print('It can be correlated. p-value:{}'.format(p))
        ttest_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        ttest_result = False

    return [p, ttest_result]    