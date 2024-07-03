from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType
import os

# openai key
#sk-6Z95RmehRkMkOQbT9f5a3f2f6407420eBe1eB0EcFb9d347d
api_key =""
# openai代理地址
api_base_url = ""
# api_base_url = "http://localhost:11434/v1/"

poetry_url = 'http://localhost:8003/dialogue'

# 使用的大语言模型
llm_model = 'gpt-4'

# 是否开启流式传输，默认开启
streaming = True

# 获取当前脚本的绝对路径
current_dir = os.path.abspath(__file__)

#获取当前脚本的目录
root_dir = os.path.dirname(current_dir)

# 构造古诗数据库文件的完整路径
poetry_path = os.path.join(root_dir,'db', 'poetry.db')

#构建关键词文件路径
keyword_path = os.path.join(root_dir,'keywords')

#构建向量模型文件路径
embedding_path = os.path.join(os.getcwd(),'bert_model', 'bce-embedding-base_v1')



#Such as Tang, Song, Yuan, Ming,Qing and so on.
# 古诗词文本转sql提示词模板
POETRY_TEXT_TO_SQL_TMPL = (
    "Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer."
    "You are required to use the following format, each taking one line:\n"
    "Question: Question here.\n"
    "SQLQuery: SQL Query to run.\n"
    "SQLEnd: LIMIT 3.\n"
    "Only use tables listed below.\n"
    "{schema}\n"
    "Question: {query_str}"
    "SQLQuery: SELECT title,author,content FROM poetry WHERE ...ORDER BY RANDOM() LIMIT 3;\n"
    "SQLEnd: ;"
)

# 创建古诗词文本转sql提示词模板对象
POETRY_TEXT_TO_SQL_PROMPT = PromptTemplate(
    POETRY_TEXT_TO_SQL_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)

# 古诗词合成回复提示词模板
POETRY_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "## Role:\n"
    "作为一名数字时代的古代文人，运用引人入胜的开场和情景描绘，用人性化的风格讲解。\n"
    "请用自己的语言重新组织，不要使用所给例子模板。\n"
    "## Mission:\n"
    "给定信息的输入，从查询结果中合成一个讲解回复。\n"
    "用户语句: {query_str}\n"
    "SQL语句: {sql_query}\n"
    "SQL查询结果: {context_str}\n"
    "回复: "
)
# 古诗词合成回复提示词模板对象
POETRY_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    POETRY_RESPONSE_SYNTHESIS_PROMPT_TMPL,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS_V2,
)
