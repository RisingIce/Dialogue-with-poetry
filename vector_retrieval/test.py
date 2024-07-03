from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, SQLDatabase, StorageContext, get_response_synthesizer, Prompt, \
    PromptHelper, PromptTemplate
from llama_index.core import ServiceContext
from llama_index.readers.database import DatabaseReader
import openai
import os
from sqlalchemy import (
    create_engine,  # 创建数据库引擎
    MetaData,  # 元数据对象
    Table,  # 数据表对象
    Column,  # 列对象
    String,  # 字符串类型
    Integer,  # 整数类型
    select,  # SQL SELECT 语句对象
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# 创建元数据对象
metadata_obj = MetaData()

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

template = (
    '#角色:\n'
    '专门回答古诗文相关问题的知识小助手，以文雅、知识渊博、亲切和耐心的方式回答问题，就像一位老师在课堂上与学生探讨诗文一样。\n'
    '#宗旨:\n'
    '旨在提供准确、详细的古诗文知识，帮助用户深入理解诗句含义、诗人背景和文化价值。\n'
    '当用户询问关于古诗文的问题时，要表达出对用户求知欲望的支持，并详细解答用户的疑问。请提供全面的回复，包括诗句的解读、诗人的历史背景、作品的文化意义、以及诗歌的艺术特色等。\n'
    '例子1：您询问的这句诗，出自诗人xxx的作品《xxx》。这句诗表达了诗人的...情感，其使用了...的修辞手法。通常来说，这首诗在当时的历史背景中代表了...。为了更好地理解这首诗，建议您可以阅读...。\n'
    '例子2：这部作品是诗人xxx在...时期创作的，它主要反映了...的社会现状。诗中的...是一个典型的文学象征，代表了...。为了深入探讨这首诗的内涵，建议您参考...。\n'
    '例子3：关于您提问的这首诗，它描绘了...的景象，展现了诗人的...情怀。这首诗的结构采用了...，语言风格上...，具有很高的艺术价值。\n'
    '根据问题用自己的语言组织回复，确保回复全面、信息丰富。\n'
    '只输出你的回复，不要输出用户的问句\n'
    '#上下文：\n'
    '{context_str}\n'
    '#用户问句：\n'
    '{query_str}\n'
)

# 创建Prompt对象
qa_template = Prompt(template)

openai.api_key = 'sk-6Z95RmehRkMkOQbT9f5a3f2f6407420eBe1eB0EcFb9d347d'
openai.base_url = "http://192.168.1.69:3001/v1/"

# 获取项目路径
current_dir = os.getcwd()

# 构造古诗数据库文件的完整路径
model_gte = os.path.join(current_dir, 'bert_model', 'gte-base-zh')
poetry_path = os.path.join(current_dir, 'db', 'poetry.db')
index_path = os.path.join(current_dir, 'vector_index')

poetry_engine = create_engine(f"sqlite:///{poetry_path}")

sql_database = SQLDatabase(poetry_engine, include_tables=["poetry"], max_string_length=1000)

llm = OpenAI(temperature=0.7, model="gpt-4", streaming=True)
embed_model = HuggingFaceEmbedding(
    model_name=model_gte
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=2048)

from llama_index.core.schema import Document

# documents = SimpleDirectoryReader(db_path, encoding='utf-8').load_data()
document = DatabaseReader(
    sql_database, poetry_engine
).load_data(query="select title,author,dynasty,content from poetry where author = '李白' ORDER BY RANDOM() LIMIT 10;")

# print(document)
index = VectorStoreIndex.from_documents(
    document, service_context=service_context
)
# index.storage_context.persist(index_path)

query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=5,
    text_qa_template=qa_template,
    refine_template=qa_template,
    summary_template=qa_template,
    simple_template=qa_template,
)


def test():
    ans = query_engine.query("讲讲李白")

    ans.print_response_stream()
    # print(ans)


#
#
if __name__ == '__main__':
    test()
#     # pass
