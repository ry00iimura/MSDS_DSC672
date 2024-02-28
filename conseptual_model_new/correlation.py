"""
Group 4

Ryosuke Iimura, DePaul University, School of Computing, RIIMURA@depaul.edu 
"""

# load libraries
import numpy as np
import pandas as pd
import scipy as sp

def pearson(arr1,arr2, print_):
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

    if print_ == True:
        print('-----------------')
        print('Pearson"s correlation :{0} p-value:{1}'.format(round(corr,3),round(pvalue,3)))
        print('arr1:{}'.format(arr1.name))
        print('arr2:{}'.format(arr2.name))
        print('-----------------')

    return corr, pvalue, 'Peason'

def spearman(arr1,arr2,print_):
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

    if print_ == True:
        print('-----------------')
        print('Spearman"s correlation:{0} p-value:{1}'.format(round(corr,3),round(pvalue,3)))
        print('arr1:{}'.format(arr1.name))
        print('arr2:{}'.format(arr2.name))
        print('-----------------')

    return corr, pvalue , 'Spearman'

def cramers_v(arr1, arr2, print_):
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
    table = np.array(pd.crosstab(arr1, arr2))
    
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

    if print_ == True:
        print('-----------------')
        print('Cramers V :{0}'.format(round(corr,3)))
        print('arr1:{}'.format(arr1.name))
        print('arr2:{}'.format(arr2.name))
        print('-----------------')
    
    return corr, np.NaN, 'Cramers V'

def corr_ratio(arr1,arr2,print_):
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

    if print_ == True:
        print('-----------------')
        print('compute correlation ratio:{}'.format(round(corr,3)))
        print('arr1:{}'.format(arr1.name))
        print('arr2:{}'.format(arr2.name))
        print('-----------------')

    return corr, np.NaN, 'correlation ratio'


def compute_correlation(arr1,arr2,arr_types,arr_scales,print_):
    """
    Decides which correlation measure is appropriate based on the types and scales of two arrays.
    
    Parameters:
    arr1 : array_like
        The first array.
    arr2 : array_like
        The second array.
    likert_scale : list
        A list of values representing the Likert scale.

    Returns:
    str
        The name of the appropriate correlation measure.
    """
    
    if arr_types ==['qualified','qualified']:

        if arr_scales == ['nominal scale','rank scale']:

            return spearman(arr1,arr2,print_)

        elif arr_scales == ['rank scale','nominal scale']:

            return spearman(arr1,arr2,print_)

        elif arr_scales == ['rank scale','rank scale']:
            
            return spearman(arr1,arr2,print_)

        elif arr_scales == ['nominal scale','nominal scale']:

            return cramers_v(arr1, arr2, print_)
        
        else:
            return np.NaN,np.NaN, 'Bad request'

    elif arr_types == ['qualified','quantified']:
        
        if arr_scales[0] == 'nominal scale' and arr_scales[1] == 'ratio scale/ interval scale':

            return corr_ratio(arr1,arr2,print_)
        
        elif arr_scales[0] == 'nominal scale' and arr_scales[1] == 'rank scale':

            return spearman(arr1,arr2,print_)
        
        elif arr_scales[0] == 'rank scale' and arr_scales[1] == 'rank scale':

            return spearman(arr1,arr2,print_)

        elif arr_scales[0] == 'rank scale' and arr_scales[1] == 'ratio scale/ interval scale':

            return spearman(arr1,arr2,print_)

        else:
            return np.NaN,np.NaN, 'Bad request'

    elif arr_types == ['quantified','qualified']:
        
        if arr_scales[0] == 'ratio scale/ interval scale' and arr_scales[1] == 'nominal scale':

            return corr_ratio(arr2,arr1,print_)
        
        elif arr_scales[0] == 'rank scale' and arr_scales[1] == 'nominal scale':

            return spearman(arr1,arr2,print_)
        
        elif arr_scales[0] == 'rank scale' and arr_scales[1] == 'rank scale':

            return spearman(arr1,arr2,print_)

        elif arr_scales[0] == 'ratio scale/ interval scale' and arr_scales[1] == 'rank scale':

            return spearman(arr1,arr2,print_)

        else:
            return np.NaN,np.NaN, 'Bad request'

    elif arr_types == ['quantified','quantified']:

        if arr_scales[0] == 'rank scale' or arr_scales[1] == 'rank scale':

            return spearman(arr1,arr2,print_)

        else:
    
            return pearson(arr1,arr2, print_)
    
    else:
        return np.NaN,np.NaN, 'Bad request'
