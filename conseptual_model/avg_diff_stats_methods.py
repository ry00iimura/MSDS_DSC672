# load libraries
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import scipy as sp

def shapiro(observed,threshold = .05):
    """
    Performs the Shapiro-Wilk test for normality on the observed data.

    Parameters:
    observed : array_like
        The observed data samples.
    threshold : float, optional
        The significance level to determine if the data is normally distributed (default is .05).

    Returns:
    list
        A list containing the p-value and a boolean indicating whether the data can be considered normally distributed.
    """
    if len(observed) < 3:
        p = 1
        shapiro_result = False
        print('shapiro result ???????????????????{}'.format(observed))

    else:
        s,p = sp.stats.shapiro(observed)

        if p <= threshold:
            print('The data can be said to normally distributed. p-value:{}'.format(p))
            shapiro_result = True
        else:
            print('The null hypothesis cannot be rejected.')
            shapiro_result = False

    return [p, shapiro_result]

def shapiro_all(arr_list,threshold):

    """
    Applies the Shapiro-Wilk test for normality to each array in a list with Bonferroni correction.

    Parameters:
    arr_list : list of array_like
        A list of data samples to be tested.
    threshold : float
        The significance level for the tests.

    Returns:
    bool
        A boolean indicating whether all the data samples can be considered normally distributed.
    """

    bonferroni_correction = threshold / len(arr_list)

    shapiro_p = []

    for arr in arr_list:
        shapiro_res = shapiro(arr,bonferroni_correction)

        shapiro_p.append(shapiro_res)

    true_ratio = np.array(shapiro_p)[:,1].sum() / len(np.array(shapiro_p)[:,1])

    if true_ratio == 1:
        shapiro_test_result = True
    else:
        shapiro_test_result = False

    return shapiro_test_result

def bartlett(threshold,*observed):

    """
    Performs Bartlett's test to assess the homogeneity of variances across samples.

    Parameters:
    threshold : float
        The significance level to determine if the variances are equal.
    *observed : multiple array_like
        Variable length argument list of arrays representing the data samples.

    Returns:
    list
        A list containing the p-value and a boolean indicating whether the samples have equal variances.
    """

    s,p = sp.stats.bartlett(*observed)

    if p <= threshold:
        print('The cannot be said that the samples are from the population with equal variance. p-value:{}'.format(p))
        bartlett_result = False
    else:
        print('The can be said that the samples are from the population with equal variance. p-value:{}'.format(p))
        bartlett_result = True

    return [p, bartlett_result]    

def one_way_ANOVA(threshold,*observed):

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
        print('The can be said that two or more groups does not have the same population mean. p-value:{}'.format(p))
        ANOVA_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        ANOVA_result = False

    return [p, ANOVA_result]    

def kruskal(threshold,*observed):

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
        print('The can be said that the population median of all of the groups are not equal. p-value:{}'.format(p))
        kruskal_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        kruskal_result = False

    return [p, kruskal_result]

def avg_diff_test(threshold,*observed):

    """
    A comprehensive test to evaluate the correlation between variables considering their distribution and variance.

    Parameters:
    threshold : float
        The significance level for the initial normality and homogeneity tests.
    *observed : multiple array_like
        Variable length argument list of arrays representing the data samples.

    Returns:
    bool
        A boolean indicating whether the variables can be considered correlated based on the tests performed.
    """

    shapiro_test_result = shapiro_all(observed,threshold)

    if shapiro_test_result == True:

        bartlett_test_result = bartlett(threshold,*observed)

        if bartlett_test_result == True:

            all_test_result = one_way_ANOVA(threshold, **observed)

        else:

            all_test_result = kruskal(threshold,*observed)

    else:

        all_test_result = kruskal(threshold,*observed)

    if all_test_result == True:

        print('The variables can be correlated')

    else:

        print('The variables would not be correlated')

    return all_test_result