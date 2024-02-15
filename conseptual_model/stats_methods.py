# load libraries
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import scipy as sp
import avg_diff_stats_methods as ad
import crosstab_stats_methods as crs
import quantified_stats_methods as qs
import correlation_stats_methods as cs

def istype(arr):
    """
    Determines if an array is quantified (numeric) or qualified (categorical).
    
    Parameters:
    arr : array_like
        The array to check.

    Returns:Pea
    str
        'quantified' if the array is numeric, 'qualified' if categorical, and 'unknown' otherwise.
    """
    if arr.dtype in [np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.float16,np.float32,np.float64]:
        return 'quantified'

    elif arr.dtype in [object,bool]:
        return 'qualified'

    else:
        return 'unknown'
    

def isscale(arr,likert_scale):
    """
    Determines if an array is a 'rank scale' or 'nominal scale' based on a given Likert scale.
    
    Parameters:
    arr : array_like
        The array to check.
    likert_scale : list
        A list of values representing the Likert scale.

    Returns:
    str
        'rank scale' if the array matches the Likert scale, 'nominal scale' otherwise.
    """
    if len(set(arr) - set(likert_scale)) == 0:
        return 'rank scale'
    else:
        return 'nominal scale'


def which_test(arr1,arr2):
    """
    Decides which statistical test is appropriate based on the types of two arrays.
    
    Parameters:
    arr1 : array_like
        The first array.
    arr2 : array_like
        The second array.

    Returns:
    tuple
        A tuple containing the names of the appropriate test and data handling method.
    """
    
    types = tuple(istype(arr) for arr in [arr1,arr2])

    if types == tuple(['qualified','qualified']):
        
        return ('chi2','crosstab')

    elif types == tuple(['qualified','quantified']):

        return ('avg diff test','tab on arr1')

    elif types == tuple(['quantified','qualified']):

        return ('avg diff test','tab on arr2')

    else:

        return ('t test','as is')
    

def which_corr(arr1,arr2,likert_scale):
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
    types = tuple(istype(arr) for arr in [arr1,arr2])

    if types == tuple(['qualified','qualified']):

        scales = tuple(isscale(arr,likert_scale) for arr in [arr1,arr2])

        if scales == tuple(['nominal scale','rank scale']):
            return 'spearman arr2'

        elif scales == tuple(['rank scale','nominal scale']):

            return 'spearman arr1'

        elif scales == tuple(['rank scale','rank scale']):
            
            return 'spearman arr1,arr2'

        else: # nominal scale and nominal scale

            return 'cramers v'

    elif types == tuple(['qualified','quantified']):

        scale = isscale(arr1,likert_scale)
        
        if scale == 'nominal scale':

            return 'correlation ratio arr1'

        else: # rank scale

            return 'spearman arr1'

    elif types == tuple(['quantified','qualified']):

        scale = isscale(arr2,likert_scale)
        
        if scale == 'nominal scale':

            return 'correlation ratio arr2'

        else: # rank scale

            return 'spearman arr2'

    else: # quantified and quantified
        return 'pearson'
    

def main_corr(arr1,arr2,likert_scale):
    """
    Determines and executes the appropriate correlation measure for two arrays.
    
    Parameters:
    arr1 : array_like
        The first array.
    arr2 : array_like
        The second array.
    likert_scale : list
        A list of values representing the Likert scale.

    Returns:
    Various
        The result of the correlation measure applied.
    """

    corr_type = which_corr(arr1,arr2,likert_scale)

    if corr_type == 'spearman arr1':

        return cs.spearman(
            arr1.replace(likert_scale).fillna(0),
            arr2.fillna(arr2.value_counts().index[0])
            )

    elif corr_type == 'spearman arr2':

        return cs.spearman(
            arr1.fillna(arr1.value_counts().index[0]),
            arr2.replace(likert_scale).fillna(0)
            )

    elif corr_type == 'spearman arr1,arr2':

        return cs.spearman(
            arr1.replace(likert_scale).fillna(0),
            arr2.replace(likert_scale).fillna(0)
            )

    elif corr_type =='pearson':

        return cs.pearson(
            arr1.fillna(np.nanmean(arr1)),
            arr2.fillna(np.nanmean(arr2))
            )

    elif corr_type =='cramers v':

        return cs.cramers_v(arr1,arr2)

    elif corr_type =='correlation ratio arr1':

        return cs.corr_ratio(arr1,arr2)

    else: # correlation ratio arr2

        return cs.corr_ratio(arr2,arr1)
    
def array_split(qualified_val,quantified_val,qualified_val_filled,quantified_val_filled):
    """
    Splits an array into sub-arrays based on unique values of a qualified variable.
    
    Parameters:
    qualified_val : array_like
        The categorical (qualified) variable to group by.
    quantified_val : array_like
        The numeric (quantified) variable to split.

    Returns:
    list
        A list of arrays, each representing a subset of quantified values corresponding to a unique qualified value.
    """
    target_qid_two_arrs_df_filled = pd.concat([
        qualified_val.fillna(qualified_val_filled),
        quantified_val.fillna(quantified_val_filled)
        ],axis = 1)

    arr_list = []

    for uv in qualified_val.unique():
        arr_list.append(target_qid_two_arrs_df_filled.loc[target_qid_two_arrs_df_filled.loc[:,qualified_val.name] == uv,quantified_val.name].values)

    return arr_list
    

def main(q_comb,target_qids_dfs,years,threshold, likert_scale,**kwargs):
    """
    Conducts statistical tests and correlation measures on combinations of questions for given years.
    
    Parameters:
    q_comb : list
        A list of question combinations to analyze.
    target_qids_dfs : dict
        A dictionary containing data frames for different years.
    years : str
        The year to analyze.
    threshold : float
        The significance level for statistical tests.
    likert_scale : list
        A list of values representing the Likert scale.

    Returns:
    list
        A list containing the results of the statistical tests and correlation measures for each question combination.
    """

    result_list = []

    for comb_idx in q_comb:
        id = ['Response ID']
        target_qid_two_arrs_df = target_qids_dfs[years]['merged'].loc[:,id + list(comb_idx)]

        arr1 = target_qid_two_arrs_df.iloc[:,2]
        arr2 = target_qid_two_arrs_df.iloc[:,1]

        pattern_res = which_test(arr1,arr2)

        if pattern_res == ('chi2','crosstab'):

            qualified_val_filled1 = fillna_value(arr1,qualified_fill_type = kwargs.get('qualified_fill_type'))
            qualified_val_filled2 = fillna_value(arr2,qualified_fill_type = kwargs.get('qualified_fill_type'))

            arr1_filled = arr1.fillna(qualified_val_filled1)
            arr2_filled = arr2.fillna(qualified_val_filled2)

            test_result = crs.crosstab_chi2(arr1_filled,arr2_filled, threshold, values = target_qid_two_arrs_df.loc[:,id], aggfunc = 'nunique') # [p,chi2_result]

        elif pattern_res == ('avg diff test','tab on arr1'):

            qualified_val_filled = fillna_value(arr1,qualified_fill_type = kwargs.get('qualified_fill_type'))
            quantified_val_filled = fillna_value(arr2,quantified_fill_type = kwargs.get('quantified_fill_type'))
            
            arr_list = array_split(arr1,arr2,qualified_val_filled,quantified_val_filled)
            test_result = ad.avg_diff_test(threshold,*arr_list)

        elif pattern_res == ('avg diff test','tab on arr2'):

            qualified_val_filled = fillna_value(arr2,qualified_fill_type = kwargs.get('qualified_fill_type'))
            quantified_val_filled = fillna_value(arr1,quantified_fill_type = kwargs.get('quantified_fill_type'))

            arr_list = array_split(arr2,arr1,qualified_val_filled,quantified_val_filled)
            test_result = ad.avg_diff_test(threshold,*arr_list)
        else:

            quantified_val_filled1 = fillna_value(arr1,quantified_fill_type = kwargs.get('quantified_fill_type'))
            quantified_val_filled2 = fillna_value(arr2,quantified_fill_type = kwargs.get('quantified_fill_type'))

            arr1_filled = arr1.fillna(quantified_val_filled1)
            arr2_filled = arr2.fillna(quantified_val_filled2)

            test_result = qs.t_test(arr1,arr2,'welch_t')

        corr_result = main_corr(arr1,arr2,likert_scale)

        result_list.append(list(comb_idx) + test_result + list(corr_result))

    return result_list


def fillna_value(arr,**kwargs):

    if kwargs.get('qualified_fill_type') == 'idxmax':

        filled_val = arr.value_counts().idxmax()

    elif kwargs.get('qualified_fill_type') == 'empty':

        filled_val = ''

    elif kwargs.get('qualified_fill_type') == 'custom':

        filled_val = kwargs.get('filled_val')

    elif kwargs.get('quantified_fill_type') == 'mean':

        filled_val = arr.mean()

    elif kwargs.get('quantified_fill_type') =='median':

        filled_val = arr.median()

    elif kwargs.get('quantified_fill_type') == 'mode':

        filled_val = arr.mode()

    elif kwargs.get('quantified_fill_type') == 'zero':

        filled_val = 0

    elif kwargs.get('quantified_fill_type') == 'custom':

        filled_val = kwargs.get('filled_val')

    return filled_val