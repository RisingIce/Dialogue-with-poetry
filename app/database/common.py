import os
import pickle
import re

import openai
from llama_index.core import (ServiceContext, SQLDatabase)
from llama_index.llms.openai import OpenAI
from sqlalchemy import (
    create_engine,  # 创建数据库引擎
    MetaData,  # 元数据对象
    Table,  # 数据表对象
    Column,  # 列对象
    String,  # 字符串类型
    Integer  # 整数类型
)
from loguru import logger
from app.database.config import (api_key, api_base_url, llm_model, poetry_path, POETRY_TEXT_TO_SQL_PROMPT,
                                 POETRY_RESPONSE_SYNTHESIS_PROMPT, keyword_path)
from app.database.utils import NLSQLTableQueryEngine,PandasQueryEngine
from app.database.config import embedding_path
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import pandas as pd
openai.api_key = api_key
openai.base_url = api_base_url

# 创建元数据对象
metadata_obj = MetaData()

# 创建数据库引擎，使用 SQLite 数据库路径
poetry_engine = create_engine(f"sqlite:///{poetry_path}")

# 定义表名
table_name = "poetry"

# 创建故事表的表结构
story_stats_table = Table(
    table_name,
    metadata_obj,
    Column("id", Integer(), primary_key=True),
    Column("title", String()),
    Column("author", String()),
    Column("dynasty", String()),
    Column("content", String())
)

sql_database = SQLDatabase(poetry_engine, include_tables=["poetry"], max_string_length=100000)


#创建诗词text2sql检索引擎·
def get_poetry_Text2SQL_engine(streaming: bool):
    llm = OpenAI(temperature=0, model=llm_model, streaming=streaming)
    # llm = Ollama(model=llm_, temperature=0.7, streaming=streaming)
    service_context = ServiceContext.from_defaults(llm=llm)

    return NLSQLTableQueryEngine(
        service_context=service_context,
        sql_database=sql_database,
        tables=[table_name],
        text_to_sql_prompt=POETRY_TEXT_TO_SQL_PROMPT,
        response_synthesis_prompt=POETRY_RESPONSE_SYNTHESIS_PROMPT,
        streaming=streaming
    )

def get_file_engine(streaming: bool,df):
   llm = OpenAI(temperature=0, model=llm_model, streaming=streaming,max_tokens= 10000)
   service_context = ServiceContext.from_defaults(llm=llm)
   return PandasQueryEngine(df=df, verbose=False, synthesize_response=True,service_context=service_context)

#关键词匹配函数
def keyword_match(text):
    # 读取关键词文件
    if os.path.isfile(keyword_path+'/title_keywords.pkl') and os.path.isfile(keyword_path+'/dynasty_keywords.pkl') and os.path.isfile(keyword_path+'/content_keywords.pkl') and os.path.isfile(keyword_path+'/author_keywords.pkl'):
        with open(keyword_path+'/title_keywords.pkl', 'rb') as file:
            title_keywords = pickle.load(file)
        with open(keyword_path+'/dynasty_keywords.pkl', 'rb') as file:
            dynasty_keywords = pickle.load(file)
        with open(keyword_path+'/content_keywords.pkl', 'rb') as file:
            content_keywords = pickle.load(file)
        with open(keyword_path+'/author_keywords.pkl', 'rb') as file:
            author_keywords = pickle.load(file)

    text_without_brackets = title_keywords.replace('(','').replace(')','').replace('（','').replace('）','')
    print(text_without_brackets)

        # print(str(title_keywords))
    # if str(title_keywords).find(text):
    #     title_match = re.findall(title_keywords, text)
    #     print(title_match)
    if text_without_brackets.find(text):
        title_match = re.findall(text_without_brackets,text)
        print(title_match)
    # dynasty_match = re.findall(dynasty_keywords,text)
    # content_match = re.findall(content_keywords,text)

    # start2 = time.time()
    # #使用find判断是否具有该字符串位置 避免多余的全匹配
    # if str(author_keywords).find(text):
    #     author_match = re.findall(author_keywords,text)
    #     print(author_match)
    # end2 = time.time()
    # print("find速度：", end2 - start2)

    # print(author_match)
    # return {
    #     'title':title_match,
    #     # 'dynasty':dynasty_match,
    #     # 'content':content_match,
    #     # 'author':author_match
    # }






