# load libraries
import correlation_test as ct
import correlation as cor
import distribution_test as dit
import utility as u
import numpy as np

def qualified_quantified_test(distribution_variance_test_result,print_,threshold,*observed):

    if distribution_variance_test_result == [True,True]: # one-way-ANOVA
        test_result = ct.one_way_ANOVA(print_,threshold,*observed)

    elif distribution_variance_test_result == [True,False]: # kruskal
        test_result = ct.kruskal(print_,threshold,*observed)

    elif distribution_variance_test_result[0] == False: # kruskal
        test_result = ct.kruskal(print_,threshold,*observed)
    else: # something wrong
        test_result = [np.nan,np.nan]

    return test_result

def main(id,q_comb,target_qids_dfs,years,print_,threshold,likert_scale,**kwargs):
    """
    Conducts statistical tests and correlation measures on combinations of questions for given years.
    
    Parameters:
    id : list
        A list to have a ID columns name.
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

        target_qid_two_arrs_df = target_qids_dfs[years]['merged'].loc[:,id + list(comb_idx)]

        arr1 = target_qid_two_arrs_df.iloc[:,2]
        arr2 = target_qid_two_arrs_df.iloc[:,1]

        arr_types = [u.istype(arr) for arr in [arr1,arr2]]
        arr_scales = [u.isscale(arr,likert_scale) for arr in [arr1,arr2]]

        if print_ == True:
            print('::::::::::::::::::::::::::::::::')
            print('arr1 : {}'.format(arr1.name))
            print('arr2 : {}'.format(arr2.name))
            print('arr_types : {}'.format(arr_types))
            print('arr_scales : {}'.format(arr_scales))
            print('::::::::::::::::::::::::::::::::')

        if arr_types == ['qualified','qualified']: # ('chi2','crosstab')

            test_result = ct.crosstab_chi2(arr1,arr2, print_ , threshold, values = target_qid_two_arrs_df.loc[:,id], aggfunc = 'nunique') # [p,chi2_result]

        elif arr_types == ['qualified','quantified']: # ('avg diff test','tab on arr1')
            
            arr_list = u.array_split(arr1,arr2)
            distribution_variance_test_result = dit.distribution_variance_test(print_,threshold,*arr_list)
            test_result = qualified_quantified_test(distribution_variance_test_result,print_,threshold,*arr_list)

        elif arr_types == ['quantified','qualified']: # ('avg diff test','tab on arr2')

            arr_list = u.array_split(arr2,arr1)
            distribution_variance_test_result = dit.distribution_variance_test(print_,threshold,*arr_list)
            test_result = qualified_quantified_test(distribution_variance_test_result,print_,threshold,*arr_list)
        
        elif arr_types == ['quantified','quantified']: # ('t test','as is')

            test_result = ct.t_test(arr1,arr2,'welch_t',print_)

        corr_result = cor.compute_correlation(arr1,arr2,arr_types,arr_scales,print_)

        result_list.append(list(comb_idx) + test_result + list(corr_result))

    return result_list