"""
Group 4

Ryosuke Iimura, DePaul University, School of Computing, RIIMURA@depaul.edu 
"""

# load libraries
import pandas as pd
import numpy as np

def array_split(qualified_val,quantified_val):
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

    target_qid_two_arrs_df = pd.concat([qualified_val, quantified_val],axis = 1)

    arr_list = []

    for uv in qualified_val.unique():
        arr_list.append(target_qid_two_arrs_df.loc[target_qid_two_arrs_df.loc[:,qualified_val.name] == uv,quantified_val.name].values)

    return arr_list

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
    if len(set(arr) - set(likert_scale.keys())) == 0 or len(set(arr) - set(likert_scale.values())) == 0 :
        return 'rank scale'
    else:
        is_type = istype(arr)
        if is_type == 'qualified' or is_type == 'unknown':
            return 'nominal scale'
        
        else:
            return 'ratio scale/ interval scale'