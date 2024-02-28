"""
Group 4

Ryosuke Iimura, DePaul University, School of Computing, RIIMURA@depaul.edu 
"""

import openai
import os
import dotenv
import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS,Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import pandas as pd

# parameters
pdf_path_list = [
    'pdf\\Microsoft-New-Future-of-Work-Report-2022.pdf',
    # 'pdf\\pwc_us_remote_work_survey.pdf',
    # 'pdf\\work_from_home_statistics_by_generation_etc_enterpriseappstoday.pdf',
    # 'pdf\\working from home Around the Globe 2023 Report.pdf',
    'pdf\\working from home Around the World.pdf'
    ]
pdf_path = pdf_path_list[0]
query_dict = {
    'WTF':[
        'Does remote work allow people to choose when they work?',
        'Does remote work improve employee work time flexibility?',
        'Does Top Management Support have an impact on work time flexibility? Is so, how does it influence on? Does it improve work time flexibility or worsen it?',
        'Does Organization Policy have an impact on work time flexibility? Is so, how does it influence on? Does it improve work time flexibility or worsen it?',
        'Is there any external factor which may impact on employee work time flexibility?',
        'Do people feel more flexible when it comes to remote work if work time is more flexible?'
        ],
    'SOP':[
        'Does remote work society or economy improve?',
        'How does remote work impact on society or economy?',
        'Does organization performances impact on society performance?',
        'Is there any difference regarding society performance if organizations belong to this society improve or worsen remote work flexibility?'
        ]}
query =  query_dict['WTF'][0]
threshold = .5

class WfhExpert:

    # load environment variable to get API key
    def auth_api_key(self):
        env_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),'.env')
        dotenv.load_dotenv()
        openai.api_key = os.environ.get('OPENAI_API_KEY')

    # pdf loader
    def ocr(self,pdf_path):
        self.loader = PyPDFLoader(pdf_path)
        return self.loader

    # pdf spliter
    def pdf_split(self):
        pages = self.loader.load_and_split()
        return pages

    # vector store
    def indexing(self, vs_cls):
        self.index = VectorstoreIndexCreator(
            vectorstore_cls=vs_cls, # vector store types Chroma or FAISS
            embedding=OpenAIEmbeddings(), # Default
        ).from_loaders([self.loader])
        return self.index

    # query and answer 
    def chat_query(self,query):
        answer = self.index.query(query)
        return answer

    # retriever
    def retriever(self,threshold):
        '''
        If the answer quality is not durable, 
        you can adjust an answer by cut off similarity threshold
        '''
        self.retriever = self.index.vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={'score_threshold': threshold}
        )

        return self.retriever

    def prompt_engineer(self):
        # prompt engineer
        prompt_template = """[INST] <<SYS>>Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}
        <</SYS>>
        Question: {question}
        Answer in English:[/INST]"""
        self.PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return self.PROMPT

    def retrievalQA(self):
        # create retrievalQA incorporated with the retriever and prompt engineering
        chain_type_kwargs = {"prompt": self.PROMPT}

        self.qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), 
            chain_type="stuff", 
            retriever=self.retriever, 
            chain_type_kwargs=chain_type_kwargs, 
            return_source_documents=True
        )

        return self.qa

    def chat_query_retrievalQA(self,query):
        result = self.qa({"query": query})
        print(result["result"])
        print("-------------------")
        print(result["source_documents"])
        return result

if __name__ == '__main__':
    ex = WfhExpert()
    ex.auth_api_key()
    ex.ocr(pdf_path_list[1])
    ex.indexing(Chroma)
    res1 = ex.chat_query(query_dict['SOP'][0])
    ex.retriever(threshold)
    ex.prompt_engineer()
    ex.retrievalQA()
    res2 = ex.chat_query_retrievalQA(query)