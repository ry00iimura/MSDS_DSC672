# load libraries
import numpy as np
import pandas as pd
import scipy as sp

def shapiro(observed,print_,threshold = .05):
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
    if len(observed) < 3 or len(np.unique(observed)) == 1:
        p = 1
        shapiro_result = True

    else:
        s,p = sp.stats.shapiro(observed)

        if p >= threshold:
            
            shapiro_result = True
            
            if print_ == True:
                print('-----------------')
                print('Shapiro-Wilk test : The data can be said to normally distributed. p-value:{}'.format(p))
                print(observed)
                print('-----------------')
            
        else:

            shapiro_result = False

            if print_ == True:
                print('-----------------')
                print('Shapiro-Wilk test : The null hypothesis cannot be rejected.')
                print(observed)
                print('-----------------')
            
    return [p, shapiro_result]

def shapiro_all(arr_list,print_,threshold):

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
        shapiro_res = shapiro(arr,print_,bonferroni_correction)

        shapiro_p.append(shapiro_res)

    true_ratio = np.array(shapiro_p)[:,1].sum() / len(np.array(shapiro_p)[:,1])

    if true_ratio == 1:
        shapiro_test_result = True
    else:
        shapiro_test_result = False

    return shapiro_test_result

def bartlett(print_,threshold,*observed):

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

        bartlett_result = False

        if print_ == True:
            print('-----------------')
            print('Bartlett test: It cannot be said that the samples are from the population with equal variance. p-value:{}'.format(p))
            print('-----------------')
        
    else:

        bartlett_result = True

        if print_ == True: 
            print('-----------------')
            print('Bartlett test : It can be said that the samples are from the population with equal variance. p-value:{}'.format(p))
            print('-----------------')

    return [p, bartlett_result] 


def distribution_variance_test(print_,threshold,*observed):

    """
    A comprehensive test to evaluate the correlation between variables considering their distribution and variance.

    Parameters:
    threshold : float
        The significance level for the initial normality and homogeneity tests.
    *observed : multiple array_like
        Variable length argument list of arrays representing the data samples.

    Returns:
    bool list
        A boolean indicating the shapiro and bartlett test results.
    """

    shapiro_test_result = shapiro_all(observed,print_,threshold)

    if shapiro_test_result == True:

        bartlett_test_result = bartlett(print_,threshold,*observed)

        test_result = [shapiro_test_result, bartlett_test_result]

    elif shapiro_test_result == False:

        test_result = [shapiro_test_result, np.nan]

    else:

        test_result = [np.nan, np.nan]

    return test_result

