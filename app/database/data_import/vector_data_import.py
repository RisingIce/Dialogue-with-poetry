from llama_index.core import SimpleDirectoryReader, SQLDatabase, StorageContext, get_response_synthesizer, Prompt, \
    PromptHelper, PromptTemplate
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from BCEmbedding.tools.llama_index import BCERerank
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import ServiceContext
from llama_index.readers.database import DatabaseReader
from llama_index.core import Document
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

vector_store = MilvusVectorStore(
    dim=768,
    collection_name="poetry",
    uri="http://124.71.22.72:19530",
    overwrite=True
)



model_base = os.path.join(current_dir, 'bert_model', 'bce-embedding-base_v1')
model_rank = os.path.join(current_dir, 'bert_model', 'bce-reranker-base_v1')
poetry_path = os.path.join(current_dir, 'app', 'database', 'db', 'poetry.db')
documents_path = os.path.join(current_dir, 'app', 'database', 'documents_data')

embed_args = {'model_name': model_base, 'max_length': 512, 'embed_batch_size': 32, 'device': 'cuda:3'}
embed_model = HuggingFaceEmbedding(**embed_args)

reranker_args = {'model': model_rank, 'top_n': 5, 'device': 'cuda:3'}
reranker_model = BCERerank(**reranker_args)

llm = OpenAI(temperature=0.7, model="gpt-4", streaming=True)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=2048)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


with open(documents_path + "/documents_data.pkl", "rb") as file:
    document = pickle.load(file)

docs = [
    Document(
        text=t['content'],
        metadata={
            'title': t['title'],
            "author": t['author'] if t['author'] != None else '',
            "dynasty": t['dynasty'] if t['dynasty'] != None else '',
        },
        metadata_seperator="/n/n",
        metadata_template="{key}:{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    ) for t in document]

# print(docs)


index = VectorStoreIndex.from_documents(
    docs, service_context=service_context, storage_context=storage_context
)