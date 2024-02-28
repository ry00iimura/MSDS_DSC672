"""
Group 4

Ryosuke Iimura, DePaul University, School of Computing, RIIMURA@depaul.edu 
"""

# load the libraries
import itertools
import pandas as pd
import numpy as np
import config_operation as co
import a_data_process as a
import b_stats_approach as b
import c_viz as c
from langchain_community.vectorstores import FAISS,Chroma
import d_chatgpt as d


# parameters
file_paths= ['dataset\\2020_rws.csv','dataset\\2021_rws.csv']
ym = co.YamlManager('parameters.yaml')
questions = ym.read_yaml()
ym2 = co.YamlManager('scales.yaml')
likert_scale = ym2.read_yaml()
threshold = d.threshold



# read the dataset
def load_raw_data(file_paths):

    rawdata_list = []

    for fp in file_paths:
        rawdata = a.D(fp,'cp1252')
        data = rawdata.del_white_space_col(inplace = True)
        rawdata_list.append(data)

    return rawdata_list[0],rawdata_list[1]

def main_process(df,likert_scale,missing_allowance = .1,fill_qualified_constant_value = 'Empty'):

    df = a.D.replacement(df,likert_scale)
    df = a.D.main_impute(df,missing_allowance,fill_qualified_constant_value)
    return df


class main(a.Q):

    def __init__(self,target_data_models,questions,questions_2020,questions_2021,rws_2020,rws_2021):

        """
        Initializes the main object by loading data.

        Parameters:
        -----------
        target_data_models : list
            The list should have two elements indicating the conceptual model.
            e.g ['TMS','WPB']
        questions : 
            This parameter is inherited from the parent class
        questions_2020 : 
            This parameter is inherited from the parent class
        questions_2021 : 
            This parameter is inherited from the parent class
        rws_2020 : 
            This parameter is inherited from the parent class
        rws_2021 : 
            This parameter is inherited from the parent class
        """

        self.target_data_models = target_data_models
        super().__init__(questions,questions_2020,questions_2021,rws_2020,rws_2021)

    def get_common_qs(self):

        """
        Search the common question IDs in "questions" dictionary by looking up the "target_data_models" as keys such ash "WPB", and retrieves them.
                
        Returns:
        --------
        dict
            A list containing the common questions ID pairs regarding the designated target_data_models keys.
        """

        target_qid_model = {}

        for i in self.target_data_models:
            target_qid_model[i] = self.get_qid_model(i,self.questions,print_ = True)

        common_q_comb = list(itertools.product(target_qid_model[self.target_data_models[0]],target_qid_model[self.target_data_models[1]]))

        return common_q_comb
    
    def execute(self,common_q_comb,likert_scale,y, printing):

        """
        Retrieves a dictionary containing six Data Frames based each common question ID --> target_qids_dfs
        Create pairs of original questions --> q_comb
        Retrieves results in the statistic tests --> result_list
                
        Returns: 
        --------
        DataFrame
            A df containing the results in the statistic test
        """

        collection = []

        for cqm in common_q_comb:

            if cqm[0] != cqm[1]:

                target_qids_dfs = self.two_qids_dfs(cqm[0],cqm[1]) # get a dict containing six DataFrames related to the original question columns and Response ID
                q_comb = self.question_combination('2021',target_qids_dfs,suffix = '_duplicated') # create pairs of original questions

                result_list = b.main(id = self.id,q_comb = q_comb,target_qids_dfs = target_qids_dfs,years = y,print_ = printing,threshold=.05,likert_scale=likert_scale) # carry out the statistic approach
                result_df = pd.DataFrame(result_list,columns = ['question1','question2','test_p-value','test_result','corr','corr_p-value','corr_test'])
                collection.append(result_df)

        collection_df = pd.concat(collection)
        collection_df = collection_df.drop_duplicates()

        return collection_df

    def add_info(self,collection_df,conceptual_model_questions):

        """
        Put the conceptual model information like "PWB" to collection_df
        Each model is stored in 'is_q1' and 'is_q2' columns.
        Even though a original question is assigned a common question ID corresponding to a specific conceptual model like "INF"(parameters.yaml), 
        it is not necessarily to get it to be categorized into the specific conceptual model (conceptual_models.yaml)

         Parameters:
        -----------
        collection_df : DataFrame
            The df should have the result in the statistic test
        conceptual_model_questions : dict
            The dict's key is the conceptual model like 'WPB', where as the value is a list of original questions
                
        Returns: 
        --------
        DataFrame
            A df containing the results in the statistic test with the additional columns
        """
        
        is_left = []
        is_right = []

        collection_df_copy = collection_df.copy()

        for idx in range(collection_df_copy.shape[0]):

            if collection_df_copy.iloc[idx,0] in conceptual_model_questions[self.target_data_models[0]]:
                is_left.append(self.target_data_models[0])
            else:
                is_left.append(np.nan)

            if collection_df_copy.iloc[idx,1] in conceptual_model_questions[self.target_data_models[1]]:
                is_right.append(self.target_data_models[1])
            else:
                is_right.append(np.nan)
            
        collection_df_copy.insert(collection_df_copy.shape[1],'is_q1',is_left)
        collection_df_copy.insert(collection_df_copy.shape[1],'is_q2',is_right)

        return collection_df_copy
    
    def get_top_corr(self,collection_df_copy,corr_strength = .3):

        """
        Retrieve the top records based on the correlation strength.
 
        Parameters:
        -----------
        collection_df_copy : DataFrame
            The df should have the result and additional columns by getting with add_info() in the statistic test
        corr_strength : float
            This is a threshold for the correlation strength. The default value is 0.3.
                
        Returns: 
        --------
        DataFrame
            A df containing the results filtered with the correlation strength 
        """

        query_txt = 'is_q1 == "{}" & is_q2 == "{}" & test_result == True & corr >= {}'.format(self.target_data_models[0],self.target_data_models[1],corr_strength)

        main_result = collection_df_copy.query(query_txt).sort_values(by = ['corr'],ascending = False) 

        return main_result
    
def chatgpt(pdf_path,query,threshold):
    ex = d.WfhExpert()
    ex.auth_api_key()
    ex.ocr(pdf_path)
    ex.indexing(Chroma)
    res1 = ex.chat_query(query)
    ex.retriever(threshold)
    ex.prompt_engineer()
    ex.retrievalQA()
    res2 = ex.chat_query_retrievalQA(query)
    return res1, res2