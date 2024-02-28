"""
Group 4

Ryosuke Iimura, DePaul University, School of Computing, RIIMURA@depaul.edu 
"""

# load libraries
import pandas as pd
import numpy as np
import itertools
from sklearn.impute import SimpleImputer
import utility as u
import config_operation as co

class D:

    """
    A class to manage and manipulate data loaded from a CSV file.

    Attributes:
    -----------
    df : DataFrame
        A pandas DataFrame containing the data loaded from the specified CSV file.

    Methods:
    --------
    del_white_space_col(inplace):
        Renames DataFrame columns by removing leading and trailing whitespace.

    get_questions_as_df():
        Returns a DataFrame containing the column names (excluding the first column) as questions.
    """
     
    def __init__(self,file_path,encode):

        """
        Initializes the D object by loading data from a CSV file into a pandas DataFrame.

        Parameters:
        -----------
        file_path : str
            The file path to the CSV file to be loaded.
        encode : str
            The encoding of the CSV file.
        """

        self.df = pd.read_csv(file_path,encoding = encode)

    def del_white_space_col(self, inplace):

        """
        Renames DataFrame columns by removing leading and trailing whitespace.

        Parameters:
        -----------
        inplace : bool
            If True, modifies the DataFrame in place. Otherwise, returns a modified copy.

        Returns:
        --------
        DataFrame or None
            Returns the modified DataFrame if inplace is False. Otherwise, modifies the DataFrame in place and returns None.
        """

        if inplace == True:

            df = self.df.rename(columns = dict(zip(self.df.columns,[c.strip() for c in self.df.columns])))

            return df

        else:
    
            return self.df.rename(columns = dict(zip(self.df.columns,[c.strip() for c in self.df.columns])))
    
    @staticmethod
    def replacement(df,replace_dict):

        df_copy = df.copy()

        for c in df_copy.columns:

            df_copy.loc[:,c].replace(replace_dict,inplace = True)
        
        return df_copy

    @staticmethod
    def compute_missing_ratio(arr):
        populations = len(arr)
        missings = arr.isna().sum()
        missing_ratio = missings/populations
        return missing_ratio

    @staticmethod
    def compute_outlier_ratio(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        outliers = arr[(arr < mean - 3 * std) | (arr > mean + 3 * std)]
        outlier_fraction = len(outliers) / len(arr)

        return outlier_fraction

    @staticmethod
    def missing_impute(arr,missing_allowance,fill_qualified_constant_value):

        arr_type = u.istype(arr)
        missing_ratio = D.compute_missing_ratio(arr)

        if arr_type == 'qualified':

            if missing_ratio <= missing_allowance:
                method = 'most_frequent'
                
            else:
                method = 'constant'

        elif arr_type == 'quantified':

            if missing_ratio <= missing_allowance:
                method = 'mean'
            else:

                outlier_fraction = D.compute_outlier_ratio(arr)

                if outlier_fraction >= .003: # 0.3%
                    method = 'median'
                else:
                    method = 'mean'

        imputer = SimpleImputer(strategy=method ,fill_value = fill_qualified_constant_value)
    
        arr_reshaped = arr.values.reshape(-1, 1)

        arr_imputed = imputer.fit_transform(arr_reshaped)

        if method == 'mean' or method == 'median':
            
            arr_imputed = np.round(arr_imputed,0)

        return arr_imputed

    @staticmethod
    def main_impute(df,missing_allowance,fill_qualified_constant_value):

        df_copy = df.copy()

        for c in df_copy.columns:

            arr = df_copy.loc[:,c]

            arr_imputed = D.missing_impute(arr,missing_allowance,fill_qualified_constant_value)

            df_copy.loc[:,c] = arr_imputed

        return df_copy

    @staticmethod    
    def get_questions_as_df(df):

        """
        Returns a DataFrame containing the column names (excluding the first column) as questions.

        Returns:
        --------
        DataFrame
            A DataFrame where each row represents a question (column name) from the original DataFrame, excluding the first column.
        """

        return pd.DataFrame(df.columns[1:], index = [i for i in range(len(df.columns[1:]))],columns = ['question'])
    


class Q:
    
    """
    A class used to manage and analyze survey question data across different years.

    Attributes:
    -----------
    id : list
        A list containing the identifier for response ID column.
    questions : dict
        A dictionary mapping question IDs to their details.
    questions_2020 : DataFrame
        A DataFrame containing the question details specific to the year 2020.
    questions_2021 : DataFrame
        A DataFrame containing the question details specific to the year 2021.
    df_2020 : DataFrame
        A DataFrame containing the survey data for the year 2020.
    df_2021 : DataFrame
        A DataFrame containing the survey data for the year 2021.

    Methods:
    --------
    get_q_info(search_qid):
        Retrieves and prints the information for a specified question ID.
        
    get_qid_type(q_type, questions, print_):
        Returns a list of question IDs of a specified type.
        
    get_qid_model(q_model, questions, print_):
        Returns a list of question IDs for a specified data model.
        
    one_qid_dfs(qid):
        Returns a list of DataFrames for a specified question ID for both years.
        
    two_qids_dfs(qid1, qid2, suffixes_):
        Merges DataFrames of two specified question IDs for both years and returns a dictionary containing them.
        
    question_combination(year, df_dict, suffix):
        Returns a list of combinations of question pairs from merged DataFrames for a specified year.
    """

    def __init__(self,questions,questions_2020,questions_2021,df_2020,df_2021):

        """
        Initializes the Q object with survey questions, their details, and survey data for years 2020 and 2021.
        
        Parameters:
        -----------
        questions : dict
            A dictionary mapping question IDs to their details.
        questions_2020 : DataFrame
            A DataFrame containing the question details specific to the year 2020.
        questions_2021 : DataFrame
            A DataFrame containing the question details specific to the year 2021.
        df_2020 : DataFrame
            A DataFrame containing the survey data for the year 2020.
        df_2021 : DataFrame
            A DataFrame containing the survey data for the year 2021.
        """

        self.id = ['Response ID']

        self.questions = questions
        
        self.questions_2020 = questions_2020
        self.questions_2021 = questions_2021

        self.df_2020 = df_2020
        self.df_2021 = df_2021

    def get_q_info(self,search_qid):

        """
        Retrieves and prints information for a specified question ID.
        
        Parameters:
        -----------
        search_qid : str
            The question ID to search for.
        
        Returns:
        --------
        dict
            A dictionary containing the information of the specified question ID.
        """

        self.q_info = self.questions[search_qid]
        self.qid_2020 = self.q_info['original IDs'][0] # question ids in 2020
        self.qid_2021 = self.q_info['original IDs'][1] # question ids in 2021
        self.q_text = self.q_info['question'] # brief question
        self.q_type = self.q_info['type']
        self.q_model = self.q_info['data model']

        return self.q_info

    def get_qid_type(self,q_type,questions,print_):

        """
        Returns a list of question IDs of a specified type.
        
        Parameters:
        -----------
        q_type : str
            The type of question to filter by.
        questions : dict
            The dictionary of questions to search within.
        print_ : bool
            Whether to print the result.
        
        Returns:
        --------
        list
            A list of question IDs matching the specified type.
        """

        qid_list = []

        for qid, q_info in questions.items():
            if q_info['type'] == q_type:
                qid_list.append(qid)

        if print_ == True:
            print('The question numbers: {}'.format(qid_list))

        return qid_list
    
    def get_qid_model(self,q_model,questions,print_):

        """
        Returns a list of question IDs for a specified data model.
        
        Parameters:
        -----------
        q_model : str
            The data model to filter by.
        questions : dict
            The dictionary of questions to search within.
        print_ : bool
            Whether to print the result.
        
        Returns:
        --------
        list
            A list of question IDs associated with the specified data model.
        """

        qid_list = []

        for qid, q_info in questions.items():
            if q_model in q_info['data model']:
                qid_list.append(qid)

        if print_ == True:
            print('The question numbers of {}: {}'.format(q_model,qid_list))

        return qid_list
    
    def one_qid_dfs(self,qid):

        """
        Returns a list of DataFrames for a specified question ID for both years.
        
        Parameters:
        -----------
        qid : str
            The question ID to retrieve data for.
        
        Returns:
        --------
        list
            A list containing DataFrames for the specified question ID from both 2020 and 2021.
        """

        df_list = []

        q_info_ = self.get_q_info(qid)

        for idx, q_dict in {0:{'qs':self.questions_2020,'q_df':self.df_2020},1:{'qs':self.questions_2021,'q_df':self.df_2021}}.items():

            target_qid_original_qids = q_info_['original IDs'][idx]

            target_qid_original_qs = q_dict['qs'].loc[target_qid_original_qids,:].values.ravel().tolist()

            result_df = q_dict['q_df'].loc[:,self.id + target_qid_original_qs]

            df_list.append(result_df)

        return df_list
    
    def two_qids_dfs(self,qid1,qid2,suffixes_ = ('','_duplicated')):

        """
        Merges DataFrames of two specified question IDs for both years and returns a dictionary containing them.
        
        Parameters:
        -----------
        qid1 : str
            The first question ID.
        qid2 : str
            The second question ID.
        suffixes_ : tuple
            Suffixes to apply to overlapping columns in the merged DataFrames.
        
        Returns:
        --------
        dict
            A dictionary containing merged DataFrames for both years, keyed by year.
        """

        df_left_list = self.one_qid_dfs(qid1)
        df_right_list = self.one_qid_dfs(qid2)

        left_right_2020 = pd.merge(df_left_list[0],df_right_list[0],on = self.id, how = 'inner',suffixes = suffixes_)
        left_right_2021 = pd.merge(df_left_list[1],df_right_list[1],on = self.id, how = 'inner',suffixes = suffixes_)

        df_dict = {
            '2020':{'left':df_left_list[0], 'right':df_right_list[0],'merged':left_right_2020},
            '2021':{'left':df_left_list[1], 'right':df_right_list[1],'merged':left_right_2021}}

        return df_dict       


    def question_combination(self,year,df_dict,suffix):

        """
        Returns a list of combinations of question pairs from merged DataFrames for a specified year.
        
        Parameters:
        -----------
        year : str
            The year to generate combinations for ('2020' or '2021').
        df_dict : dict
            A dictionary containing DataFrames for the specified year.
        suffix : str
            A suffix to apply to duplicate question columns in the right DataFrame.
        
        Returns:
        --------
        list
            A list of tuples representing all possible question combinations.
        """

        left_qs = list(set(df_dict[year]['left'].columns))
        left_qs.remove(self.id[0])

        right_qs = list(set(df_dict[year]['right'].columns))
        right_qs.remove(self.id[0])
        right_qs

        duplicated_qs = set(left_qs) & set(right_qs)

        right_qs = ['{}{}'.format(q,suffix) if q in duplicated_qs else q for q in right_qs]

        return list(itertools.product(left_qs,right_qs))
    
    @staticmethod    
    def get_qdfs(rawdata_list):

        questions_2020 = D.get_questions_as_df(rawdata_list[0])
        questions_2021 = D.get_questions_as_df(rawdata_list[1])

        return questions_2020, questions_2021
    

if __name__ == '__main__':

    missing_allowance = .1
    fill_qualified_constant_value = 'empty'
    d = D('dataset\\2020_rws.csv','cp1252')
    df = d.del_white_space_col(inplace = True)
    D.get_questions_as_df(df)

    ym = co.YamlManager('scales.yaml')
    likert_scale = ym.read_yaml()
    df = D.replacement(df,likert_scale)
    arr = df.iloc[:,0]
    D.compute_missing_ratio(arr)
    D.compute_outlier_ratio(arr)
    D.missing_impute(arr,missing_allowance,fill_qualified_constant_value)
    D.main_impute(df,missing_allowance,fill_qualified_constant_value)