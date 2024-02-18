# load libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import level_correlation_test as lct
import utility as u
import distribution_test as dt
import textwrap

class viz:

    def __init__(self,idx, main_result,df,id_col):

            self.q1 = main_result.iloc[idx,0]
            self.q2 = main_result.iloc[idx,1]
            self.validation_df = df.loc[:,id_col + [self.q1,self.q2]]
            self.validation_ctab = pd.crosstab(self.validation_df[self.q1],self.validation_df[self.q2],self.validation_df[id_col[0]],aggfunc = 'nunique')

    def get_arrs(self):
         
        self.arr1 = self.validation_df.loc[:,self.q1]
        self.arr2 = self.validation_df.loc[:,self.q2]

        return self.arr1, self.arr2
    
    def get_observed(self,arr1,arr2,pattern):

        if pattern == '1':
            observed = u.array_split(arr1,arr2)
        else:
            observed = u.array_split(arr2,arr1)

        return observed
    
    def dist_var_test(self,print_,threshold,*observed):

        test_result = dt.distribution_variance_test(print_,threshold,*observed)

        return test_result

    def is_types(self,arr1,arr2):
        
        arr_types = [u.istype(arr) for arr in [arr1,arr2]]

        print(arr_types)
                     
        return arr_types
    
    def is_scales(self,arr1,arr2,likert_scale):
        
        arr_scales = [u.isscale(arr,likert_scale) for arr in [arr1,arr2]]

        print(arr_scales)
                     
        return arr_scales


    def tukey(self,arr1,arr2):

        lct.tukey(arr1,arr2)


    def heatmap(self,viz_type,size = (7,10), fsize = 8):

        if viz_type == 'count':

            plt.figure(figsize = size)
            sns.heatmap(self.validation_ctab ,annot = True)
            plt.xlabel(self.q2,fontsize = fsize,wrap = True)
            plt.ylabel(self.q1,fontsize = fsize,wrap = True)
            plt.tight_layout()

        elif viz_type == 'chi2_residual':

            self.chi2_residual_df = lct.chi2_residual(self.validation_ctab)
            self.chi2_residual_abs_df = abs(self.chi2_residual_df)
            
            plt.figure(figsize = size)
            sns.heatmap(self.chi2_residual_df.where(self.chi2_residual_abs_df > 1.96, 0),annot = True)
            plt.xlabel(self.q2,fontsize = fsize,wrap = True)
            plt.ylabel(self.q1,fontsize = fsize,wrap = True)
            plt.tight_layout()

        elif viz_type == 'both':
            # Adjust the figure size as needed to accommodate two subplots side by side
            plt.figure(figsize=(size[0]*2, size[1]))
            
            # First subplot for the count heatmap
            plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
            sns.heatmap(self.validation_ctab, annot=True)
            plt.xlabel(self.q2, fontsize=fsize, wrap=True)
            plt.ylabel(self.q1, fontsize=fsize, wrap=True)

            # Second subplot for the chi2_residual heatmap
            plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
            self.chi2_residual_df = lct.chi2_residual(self.validation_ctab)
            self.chi2_residual_abs_df = abs(self.chi2_residual_df)
            sns.heatmap(self.chi2_residual_df.where(self.chi2_residual_abs_df > 1.96, 0), annot=True)
            plt.xlabel(self.q2, fontsize=fsize, wrap=True)
            plt.ylabel(self.q1, fontsize=fsize, wrap=True)
            
        elif viz_type =='steel_dwass_type1':

            fig = plt.figure(figsize=(size[0]*2, size[1]))
            fig.suptitle('Steel Dwass test: which levels have different median on the variable of {}'.format(textwrap.fill(self.q2, width=60)))

            # First subplot for the count heatmap
            plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
            sns.heatmap(lct.steel_dwass(self.arr1,self.arr2),annot = True)
            plt.xlabel(self.q1, fontsize=fsize, wrap=True)

            # Second subplot for the chi2_residual heatmap
            plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
            sns.heatmap(lct.steel_dwass(self.arr1,self.arr2) < .05, annot=True)
            plt.xlabel(self.q1, fontsize=fsize, wrap=True)
            
            plt.tight_layout()

        elif viz_type =='steel_dwass_type2':

            fig =plt.figure(figsize=(size[0]*2, size[1]))
            fig.suptitle('Steel Dwass test: which levels have different median on the variable of {}'.format(textwrap.fill(self.q1, width=60)))

            # First subplot for the count heatmap
            plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
            sns.heatmap(lct.steel_dwass(self.arr2,self.arr1),annot = True)
            plt.xlabel(self.q2, fontsize=fsize, wrap=True)

            # Second subplot for the chi2_residual heatmap
            plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
            sns.heatmap(lct.steel_dwass(self.arr2,self.arr1) < .05, annot=True)
            plt.xlabel(self.q2, fontsize=fsize, wrap=True)
            
            plt.tight_layout()

