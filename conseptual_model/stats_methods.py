# load libraries
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import scipy as sp

"""
 statistical tests
"""

def t_test(arr1,arr2,test_type,threshold = .05):
    '''
    Student T test 
    '''
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


def chi2(observed,threshold = .05):
    chi2, p, dof, expected = chi2_contingency(observed)

    if p <= threshold:
        print('It can be correlated. p-value:{}'.format(p))
        chi2_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        chi2_result = False

    return [p, chi2_result]

def shapiro(observed,threshold = .05):
    s,p = sp.stats.shapiro(observed)

    if p <= threshold:
        print('The data can be said to normally distributed. p-value:{}'.format(p))
        shapiro_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        shapiro_result = False

    return [p, shapiro_result]

def shapiro_all(arr_list,threshold):

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

    s,p = sp.stats.bartlett(*observed)

    if p <= threshold:
        print('The cannot be said that the samples are from the population with equal variance. p-value:{}'.format(p))
        bartlett_result = False
    else:
        print('The can be said that the samples are from the population with equal variance. p-value:{}'.format(p))
        bartlett_result = True

    return [p, bartlett_result]    

def one_way_ANOVA(threshold,*observed):

    s , p = sp.stats.f_oneway(*observed)

    if p <= threshold:
        print('The can be said that two or more groups does not have the same population mean. p-value:{}'.format(p))
        ANOVA_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        ANOVA_result = False

    return [p, ANOVA_result]    

def kruskal(threshold,*observed):

    s , p = sp.stats.kruskal(*observed)

    if p <= threshold:
        print('The can be said that the population median of all of the groups are not equal. p-value:{}'.format(p))
        kruskal_result = True
    else:
        print('The null hypothesis cannot be rejected.')
        kruskal_result = False

    return [p, kruskal_result] 

"""
correlation power computation
"""

def pearson(arr1,arr2):
    corr , pvalue = sp.stats.pearsonr(arr1,arr2)
    print('Pearson"s correlation :{0} p-value:{1}'.format(round(corr,3),round(pvalue,3)))
    return corr, pvalue, 'Peason'

def spearman(arr1,arr2):
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



"""
is_* functions
"""

def istype(arr):
    '''
    return if arr is 'quantified' or 'qualified
    '''
    if arr.dtype in [np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.float16,np.float32,np.float64]:
        return 'quantified'

    elif arr.dtype in [object,bool]:
        return 'qualified'

    else:
        return 'unknown'
    

def isscale(arr,likert_scale):
    '''
    return if arr is 'rank scale' or 'nominal scale'
    '''

    if len(set(arr) - set(likert_scale)) == 0:
        return 'rank scale'
    else:
        return 'nominal scale'


def crosstab_chi2(arr1,arr2,threshold,**kwargs):
    
    arrays = {}

    iter = 1

    for arr in [arr1,arr2]:
        arr_copy = arr.copy()
        arr_filled_value = arr_copy.value_counts().idxmax()
        arr_new = arr_copy.fillna(arr_filled_value)

        arrays[iter] = arr_new

        iter += 1

    observed = pd.crosstab(arrays[1],arrays[2],**kwargs)
    observed_filled = observed.fillna(0)

    return chi2(observed_filled,threshold)
    

def which_test(arr1,arr2):
    
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

    corr_type = which_corr(arr1,arr2,likert_scale)

    if corr_type == 'spearman arr1':

        return spearman(
            arr1.replace(likert_scale).fillna(0),
            arr2.fillna(arr2.value_counts().index[0])
            )

    elif corr_type == 'spearman arr2':

        return spearman(
            arr1.fillna(arr1.value_counts().index[0]),
            arr2.replace(likert_scale).fillna(0)
            )

    elif corr_type == 'spearman arr1,arr2':

        return spearman(
            arr1.replace(likert_scale).fillna(0),
            arr2.replace(likert_scale).fillna(0)
            )

    elif corr_type =='pearson':

        return pearson(
            arr1.fillna(np.nanmean(arr1)),
            arr2.fillna(np.nanmean(arr2))
            )

    elif corr_type =='cramers v':

        return cramers_v(arr1,arr2)

    elif corr_type =='correlation ratio arr1':

        return corr_ratio(arr1,arr2)

    else: # correlation ratio arr2

        return corr_ratio(arr2,arr1)
    
def array_split(qualified_val,quantified_val):

    qualified_val_filled = qualified_val.value_counts().idxmax()
    quantified_val_filled = quantified_val.mean()

    target_qid_two_arrs_df_filled = pd.concat([
        qualified_val.fillna(qualified_val_filled),
        quantified_val.fillna(quantified_val_filled)
        ],axis = 1)

    arr_list = []

    for uv in qualified_val.unique():
        arr_list.append(target_qid_two_arrs_df_filled.loc[target_qid_two_arrs_df_filled.loc[:,qualified_val.name] == uv,quantified_val.name].values)

    return arr_list
    

def avg_diff_test(threshold,*observed):

    shapiro_test_result = shapiro_all(observed,threshold)

    if shapiro_test_result == True:

        bartlett_test_result = bartlett(threshold,*observed)

        if bartlett_test_result == True:

            all_test_result = one_way_ANOVA(threshold, **observed)

        else:

            kruskal_test_result = kruskal(threshold,*observed)

    else:

        all_test_result = kruskal(threshold,*observed)

    if all_test_result == True:

        print('The variables can be correlated')

    else:

        print('The variables would not be correlated')

    return all_test_result


def main(q_comb,target_qids_dfs,years,threshold, likert_scale):

    result_list = []

    for comb_idx in q_comb:
        id = ['Response ID']
        target_qid_two_arrs_df = target_qids_dfs[years]['merged'].loc[:,id + list(comb_idx)]

        arr1 = target_qid_two_arrs_df.iloc[:,2]
        arr2 = target_qid_two_arrs_df.iloc[:,1]

        pattern_res = which_test(arr1,arr2)

        if pattern_res == ('chi2','crosstab'):
            test_result = crosstab_chi2(arr1,arr2, threshold, values = target_qid_two_arrs_df.loc[:,id], aggfunc = 'nunique') # [p,chi2_result]

        elif pattern_res == ('avg diff test','tab on arr1'):
            arr_list = array_split(arr1,arr2)
            test_result = avg_diff_test(threshold,*arr_list)

        elif pattern_res == ('avg diff test','tab on arr2'):
            arr_list = array_split(arr2,arr1)
            test_result = avg_diff_test(threshold,*arr_list)
        else:
            test_result = t_test(arr1,arr2,'welch_t')

        corr_result = main_corr(arr1,arr2,likert_scale)

        result_list.append(list(comb_idx) + test_result + list(corr_result))

    return result_list