from llama_index.core import (ServiceContext, SQLDatabase, BasePromptTemplate, PromptTemplate, QueryBundle, Response,
                              get_response_synthesizer
                              )
from llama_index.core.prompts import PromptType
from sqlalchemy import Table
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.query_engine import BaseQueryEngine
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from abc import abstractmethod
from llama_index.core.retrievers import NLSQLRetriever
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.objects.table_node_mapping import SQLTableSchema
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_TO_SQL_PROMPT,
)
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    embed_model_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Table
from app.database.common import logger


# llama-index默认模板
# **NOTE**: deprecated (for older versions of sql query engine)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {sql_response_str}\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS,
)

# **NOTE**: newer version of sql query engine
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL_V2 = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_V2 = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL_V2,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS_V2,
)


# 查询引擎源码
class BaseSQLTableQueryEngine(BaseQueryEngine):
    def __init__(
            self,
            synthesize_response: bool = True,
            response_synthesis_prompt: Optional[BasePromptTemplate] = None,
            service_context: Optional[ServiceContext] = None,
            verbose: bool = False,
            streaming: bool = True,
            **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._response_synthesis_prompt = (
                response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT_V2
        )
        self._streaming = streaming
        # do some basic prompt validation
        _validate_prompt(self._response_synthesis_prompt)
        self._synthesize_response = synthesize_response
        self._verbose = verbose
        super().__init__(self._service_context.callback_manager, **kwargs)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"response_synthesis_prompt": self._response_synthesis_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"sql_retriever": self.sql_retriever}

    @property
    @abstractmethod
    def sql_retriever(self) -> NLSQLRetriever:
        """Get SQL retriever."""

    @property
    def service_context(self) -> ServiceContext:
        """Get service context."""
        return self._service_context

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        retrieved_nodes, metadata = self.sql_retriever.retrieve_with_metadata(
            query_bundle
        )
        sql_query_str = metadata["sql_query"]
        if self._synthesize_response:
            partial_synthesis_prompt = self._response_synthesis_prompt.partial_format(
                sql_query=sql_query_str,
            )
            response_synthesizer = get_response_synthesizer(
                service_context=self._service_context,
                callback_manager=self._service_context.callback_manager,
                text_qa_template=partial_synthesis_prompt,
                verbose=self._verbose,
                streaming=self._streaming
            )
            response = response_synthesizer.synthesize(
                query=query_bundle.query_str,
                nodes=retrieved_nodes,
            )
            cast(Dict, response.metadata).update(metadata)
            return cast(Response, response)
        else:
            response_str = "\n".join([node.node.text for node in retrieved_nodes])
            return Response(response=response_str, metadata=metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        retrieved_nodes, metadata = await self.sql_retriever.aretrieve_with_metadata(
            query_bundle
        )

        sql_query_str = metadata["sql_query"]
        if self._synthesize_response:
            partial_synthesis_prompt = self._response_synthesis_prompt.partial_format(
                sql_query=sql_query_str,
            )
            response_synthesizer = get_response_synthesizer(
                service_context=self._service_context,
                callback_manager=self._service_context.callback_manager,
                text_qa_template=partial_synthesis_prompt,

            )
            response = await response_synthesizer.asynthesize(
                query=query_bundle.query_str,
                nodes=retrieved_nodes,
            )
            cast(Dict, response.metadata).update(metadata)
            return cast(Response, response)
        else:
            response_str = "\n".join([node.node.text for node in retrieved_nodes])
            return Response(response=response_str, metadata=metadata)


class NLSQLTableQueryEngine(BaseSQLTableQueryEngine):
    """
    Natural language SQL Table query engine.

    Read NLStructStoreQueryEngine's docstring for more info on NL SQL.
    """

    def __init__(
            self,
            sql_database: SQLDatabase,
            text_to_sql_prompt: Optional[BasePromptTemplate] = None,
            context_query_kwargs: Optional[dict] = None,
            synthesize_response: bool = True,
            response_synthesis_prompt: Optional[BasePromptTemplate] = None,
            tables: Optional[Union[List[str], List[Table]]] = None,
            service_context: Optional[ServiceContext] = None,
            context_str_prefix: Optional[str] = None,
            sql_only: bool = False,
            verbose: bool = False,
            streaming: bool = False,
            sql_query: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # self._tables = tables
        self._sql_retriever = NLSQLRetriever(
            sql_database,
            text_to_sql_prompt=text_to_sql_prompt,
            context_query_kwargs=context_query_kwargs,
            tables=tables,
            context_str_prefix=context_str_prefix,
            service_context=service_context,
            sql_only=sql_only,
            verbose=verbose,
            streaming = streaming,
            sql_query=sql_query
        )
        super().__init__(
            synthesize_response=synthesize_response,
            response_synthesis_prompt=response_synthesis_prompt,
            service_context=service_context,
            verbose=verbose,
            streaming=streaming,
            **kwargs,
        )

    @property
    def sql_retriever(self) -> NLSQLRetriever:
        """Get SQL retriever."""
        return self._sql_retriever


def _validate_prompt(response_synthesis_prompt: BasePromptTemplate) -> None:
    """Validate prompt."""
    if (
            response_synthesis_prompt.template_vars
            != DEFAULT_RESPONSE_SYNTHESIS_PROMPT_V2.template_vars
    ):
        raise ValueError(
            "response_synthesis_prompt must have the following template variables: "
            "query_str, sql_query, context_str"
        )



# # 检索器源码
# class SQLRetriever(BaseRetriever):
#     """SQL Retriever.

#     Retrieves via raw SQL statements.

#     Args:
#         sql_database (SQLDatabase): SQL database.
#         return_raw (bool): Whether to return raw results or format results.
#             Defaults to True.

#     """

#     def __init__(
#             self,
#             sql_database: SQLDatabase,
#             return_raw: bool = True,
#             callback_manager: Optional[CallbackManager] = None,
#             **kwargs: Any,
#     ) -> None:
#         """Initialize params."""
#         self._sql_database = sql_database
#         self._return_raw = return_raw
#         super().__init__(callback_manager)

#     def _format_node_results(
#             self, results: List[List[Any]], col_keys: List[str]
#     ) -> List[NodeWithScore]:
#         """Format node results."""
#         nodes = []
#         for result in results:
#             # associate column keys with result tuple
#             metadata = dict(zip(col_keys, result))
#             # NOTE: leave text field blank for now
#             text_node = TextNode(
#                 text="",
#                 metadata=metadata,
#             )
#             nodes.append(NodeWithScore(node=text_node))
#         return nodes

#     def retrieve_with_metadata(
#             self, str_or_query_bundle: QueryType
#     ) -> Tuple[List[NodeWithScore], Dict]:
#         """Retrieve with metadata."""
#         if isinstance(str_or_query_bundle, str):
#             query_bundle = QueryBundle(str_or_query_bundle)
#         else:
#             query_bundle = str_or_query_bundle
#         raw_response_str, metadata = self._sql_database.run_sql(query_bundle.query_str)
#         if self._return_raw:
#             return [NodeWithScore(node=TextNode(text=raw_response_str))], metadata
#         else:
#             # return formatted
#             results = metadata["result"]
#             col_keys = metadata["col_keys"]
#             return self._format_node_results(results, col_keys), metadata

#     async def aretrieve_with_metadata(
#             self, str_or_query_bundle: QueryType
#     ) -> Tuple[List[NodeWithScore], Dict]:
#         return self.retrieve_with_metadata(str_or_query_bundle)

#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve nodes given query."""
#         retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
#         return retrieved_nodes


# class SQLParserMode(str, Enum):
#     """SQL Parser Mode."""

#     DEFAULT = "default"
#     PGVECTOR = "pgvector"


# class BaseSQLParser(ABC):
#     """Base SQL Parser."""

#     @abstractmethod
#     def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
#         """Parse response to SQL."""


# class DefaultSQLParser(BaseSQLParser):
#     """Default SQL Parser."""

#     def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
#         """Parse response to SQL."""
#         sql_query_start = response.find("SQLQuery:")
#         if sql_query_start != -1:
#             response = response[sql_query_start:]
#             # TODO: move to removeprefix after Python 3.9+
#             if response.startswith("SQLQuery:"):
#                 response = response[len("SQLQuery:"):]
#         sql_result_start = response.find("SQLEnd:")
#         if sql_result_start != -1:
#             response = response[:sql_result_start]
#         return response.strip().strip("```").strip()


# class PGVectorSQLParser(BaseSQLParser):
#     """PGVector SQL Parser."""

#     def __init__(
#             self,
#             embed_model: BaseEmbedding,
#     ) -> None:
#         """Initialize params."""
#         self._embed_model = embed_model

#     def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
#         """Parse response to SQL."""
#         sql_query_start = response.find("SQLQuery:")
#         if sql_query_start != -1:
#             response = response[sql_query_start:]
#             # TODO: move to removeprefix after Python 3.9+
#             if response.startswith("SQLQuery:"):
#                 response = response[len("SQLQuery:"):]
#         sql_result_start = response.find("SQLEnd:")
#         if sql_result_start != -1:
#             response = response[:sql_result_start]

#         # this gets you the sql string with [query_vector] placeholders
#         raw_sql_str = response.strip().strip("```").strip()
#         query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
#         query_embedding_str = str(query_embedding)
#         return raw_sql_str.replace("[query_vector]", query_embedding_str)


# class NLSQLRetriever(BaseRetriever, PromptMixin):
#     """Text-to-SQL Retriever.

#     Retrieves via text.

#     Args:
#         sql_database (SQLDatabase): SQL database.
#         text_to_sql_prompt (BasePromptTemplate): Prompt template for text-to-sql.
#             Defaults to DEFAULT_TEXT_TO_SQL_PROMPT.
#         context_query_kwargs (dict): Mapping from table name to context query.
#             Defaults to None.
#         tables (Union[List[str], List[Table]]): List of table names or Table objects.
#         table_retriever (ObjectRetriever[SQLTableSchema]): Object retriever for
#             SQLTableSchema objects. Defaults to None.
#         context_str_prefix (str): Prefix for context string. Defaults to None.
#         service_context (ServiceContext): Service context. Defaults to None.
#         return_raw (bool): Whether to return plain-text dump of SQL results, or parsed into Nodes.
#         handle_sql_errors (bool): Whether to handle SQL errors. Defaults to True.
#         sql_only (bool) : Whether to get only sql and not the sql query result.
#             Default to False.
#         llm (Optional[LLM]): Language model to use.

#     """

#     def __init__(
#             self,
#             sql_database: SQLDatabase,
#             text_to_sql_prompt: Optional[BasePromptTemplate] = None,
#             context_query_kwargs: Optional[dict] = None,
#             tables: Optional[Union[List[str], List[Table]]] = None,
#             table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
#             context_str_prefix: Optional[str] = None,
#             sql_parser_mode: SQLParserMode = SQLParserMode.DEFAULT,
#             llm: Optional[LLM] = None,
#             embed_model: Optional[BaseEmbedding] = None,
#             service_context: Optional[ServiceContext] = None,
#             return_raw: bool = True,
#             handle_sql_errors: bool = True,
#             sql_only: bool = False,
#             callback_manager: Optional[CallbackManager] = None,
#             verbose: bool = False,
#             sql_query: Optional[str] = None,
#             **kwargs: Any,
#     ) -> None:
#         """Initialize params."""
#         self._sql_retriever = SQLRetriever(sql_database, return_raw=return_raw)
#         self._sql_database = sql_database
#         self._get_tables = self._load_get_tables_fn(
#             sql_database, tables, context_query_kwargs, table_retriever
#         )
#         self._context_str_prefix = context_str_prefix
#         self._llm = llm or llm_from_settings_or_context(Settings, service_context)
#         self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
#         self._sql_parser_mode = sql_parser_mode

#         embed_model = embed_model or embed_model_from_settings_or_context(
#             Settings, service_context
#         )
#         self._sql_parser = self._load_sql_parser(sql_parser_mode, embed_model)
#         self._handle_sql_errors = handle_sql_errors
#         self._sql_only = sql_only
#         self._verbose = verbose
#         self._sql_query = sql_query
#         super().__init__(
#             callback_manager=callback_manager
#                              or callback_manager_from_settings_or_context(Settings, service_context)
#         )

#     def _get_prompts(self) -> Dict[str, Any]:
#         """Get prompts."""
#         return {
#             "text_to_sql_prompt": self._text_to_sql_prompt,
#         }

#     def _update_prompts(self, prompts: PromptDictType) -> None:
#         """Update prompts."""
#         if "text_to_sql_prompt" in prompts:
#             self._text_to_sql_prompt = prompts["text_to_sql_prompt"]

#     def _get_prompt_modules(self) -> PromptMixinType:
#         """Get prompt modules."""
#         return {}

#     def _load_sql_parser(
#             self, sql_parser_mode: SQLParserMode, embed_model: BaseEmbedding
#     ) -> BaseSQLParser:
#         """Load SQL parser."""
#         if sql_parser_mode == SQLParserMode.DEFAULT:
#             return DefaultSQLParser()
#         elif sql_parser_mode == SQLParserMode.PGVECTOR:
#             return PGVectorSQLParser(embed_model=embed_model)
#         else:
#             raise ValueError(f"Unknown SQL parser mode: {sql_parser_mode}")

#     def _load_get_tables_fn(
#             self,
#             sql_database: SQLDatabase,
#             tables: Optional[Union[List[str], List[Table]]] = None,
#             context_query_kwargs: Optional[dict] = None,
#             table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
#     ) -> Callable[[str], List[SQLTableSchema]]:
#         """Load get_tables function."""
#         context_query_kwargs = context_query_kwargs or {}
#         if table_retriever is not None:
#             return lambda query_str: cast(Any, table_retriever).retrieve(query_str)
#         else:
#             if tables is not None:
#                 table_names: List[str] = [
#                     t.name if isinstance(t, Table) else t for t in tables
#                 ]
#             else:
#                 table_names = list(sql_database.get_usable_table_names())
#             context_strs = [context_query_kwargs.get(t, None) for t in table_names]
#             table_schemas = [
#                 SQLTableSchema(table_name=t, context_str=c)
#                 for t, c in zip(table_names, context_strs)
#             ]
#             return lambda _: table_schemas

#     def retrieve_with_metadata(
#             self, str_or_query_bundle: QueryType
#     ) -> Tuple[List[NodeWithScore], Dict]:
#         """Retrieve with metadata."""
#         if isinstance(str_or_query_bundle, str):
#             query_bundle = QueryBundle(str_or_query_bundle)
#         else:
#             query_bundle = str_or_query_bundle
#         table_desc_str = self._get_table_context(query_bundle)
#         logger.debug(f"数据表格信息: {table_desc_str}")
#         if self._verbose:
#             print(f"数据表格信息: {table_desc_str}")
#         if self._sql_query:
#             sql_query_str = self._sql_query
#         else:
#             response_str = self._llm.predict(
#                 self._text_to_sql_prompt,
#                 query_str=query_bundle.query_str,
#                 schema=table_desc_str,
#                 dialect=self._sql_database.dialect,
#             )
#             logger.debug(f"模型text2sql预测：{response_str}")

#             sql_query_str = self._sql_parser.parse_response_to_sql(
#                 response_str, query_bundle
#             )
#         logger.debug(f"解析出的SQL语句：{sql_query_str}")

#         # assume that it's a valid SQL query
#         if self._verbose:
#             print(f"> Predicted SQL query: {sql_query_str}")

#         if self._sql_only:
#             sql_only_node = TextNode(text=f"{sql_query_str}")
#             retrieved_nodes = [NodeWithScore(node=sql_only_node)]
#             metadata = {"result": sql_query_str}
#         else:
#             try:
#                 retrieved_nodes, metadata = self._sql_retriever.retrieve_with_metadata(
#                     sql_query_str
#                 )
#             except BaseException as e:
#                 # if handle_sql_errors is True, then return error message
#                 if self._handle_sql_errors:
#                     err_node = TextNode(text=f"Error: {e!s}")
#                     retrieved_nodes = [NodeWithScore(node=err_node)]
#                     metadata = {}
#                 else:
#                     raise
#             logger.debug(f"元数据：{metadata}")

#         return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

#     async def aretrieve_with_metadata(
#             self, str_or_query_bundle: QueryType
#     ) -> Tuple[List[NodeWithScore], Dict]:
#         """Async retrieve with metadata."""
#         if isinstance(str_or_query_bundle, str):
#             query_bundle = QueryBundle(str_or_query_bundle)
#         else:
#             query_bundle = str_or_query_bundle
#         table_desc_str = self._get_table_context(query_bundle)
#         logger.debug(f"数据表格信息: {table_desc_str}")

#         response_str = await self._llm.apredict(
#             self._text_to_sql_prompt,
#             query_str=query_bundle.query_str,
#             schema=table_desc_str,
#             dialect=self._sql_database.dialect,
#         )

#         sql_query_str = self._sql_parser.parse_response_to_sql(
#             response_str, query_bundle
#         )
#         # assume that it's a valid SQL query
#         logger.debug(f"预测的SQL语句: {sql_query_str}")

#         if self._sql_only:
#             sql_only_node = TextNode(text=f"{sql_query_str}")
#             retrieved_nodes = [NodeWithScore(node=sql_only_node)]
#             metadata: Dict[str, Any] = {}
#         else:
#             try:
#                 (
#                     retrieved_nodes,
#                     metadata,
#                 ) = await self._sql_retriever.aretrieve_with_metadata(sql_query_str)
#             except BaseException as e:
#                 # if handle_sql_errors is True, then return error message
#                 if self._handle_sql_errors:
#                     err_node = TextNode(text=f"Error: {e!s}")
#                     retrieved_nodes = [NodeWithScore(node=err_node)]
#                     metadata = {}
#                 else:
#                     raise
#         return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve nodes given query."""
#         retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
#         return retrieved_nodes

#     async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Async retrieve nodes given query."""
#         retrieved_nodes, _ = await self.aretrieve_with_metadata(query_bundle)
#         return retrieved_nodes

#     def _get_table_context(self, query_bundle: QueryBundle) -> str:
#         """Get table context.

#         Get tables schema + optional context as a single string.

#         """
#         table_schema_objs = self._get_tables(query_bundle.query_str)
#         context_strs = []
#         if self._context_str_prefix is not None:
#             context_strs = [self._context_str_prefix]

#         for table_schema_obj in table_schema_objs:
#             table_info = self._sql_database.get_single_table_info(
#                 table_schema_obj.table_name
#             )

#             if table_schema_obj.context_str:
#                 table_opt_context = " The table description is: "
#                 table_opt_context += table_schema_obj.context_str
#                 table_info += table_opt_context

#             context_strs.append(table_info)

#         return "\n\n".join(context_strs)
    



"""Default query for PandasIndex.

WARNING: This tool provides the LLM with access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
This tool is not recommended to be used in a production setting, and would
require heavy sandboxing or virtual machines

"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response,StreamingResponse
from llama_index.core.indices.struct_store.pandas import PandasIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.utils import print_text
from llama_index.experimental.query_engine.pandas.prompts import DEFAULT_PANDAS_PROMPT
from llama_index.experimental.query_engine.pandas.output_parser import (
    PandasInstructionParser,
)

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)


# **NOTE**: newer version of sql query engine
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)

#
class PandasQueryEngine(BaseQueryEngine):
    """Pandas query engine.

    Convert natural language to Pandas python code.

    WARNING: This tool provides the Agent access to the `eval` function.
    Arbitrary code execution is possible on the machine running this tool.
    This tool is not recommended to be used in a production setting, and would
    require heavy sandboxing or virtual machines


    Args:
        df (pd.DataFrame): Pandas dataframe to use.
        instruction_str (Optional[str]): Instruction string to use.
        output_processor (Optional[Callable[[str], str]]): Output processor.
            A callable that takes in the output string, pandas DataFrame,
            and any output kwargs and returns a string.
            eg.kwargs["max_colwidth"] = [int] is used to set the length of text
            that each column can display during str(df). Set it to a higher number
            if there is possibly long text in the dataframe.
        pandas_prompt (Optional[BasePromptTemplate]): Pandas prompt to use.
        head (int): Number of rows to show in the table context.
        llm (Optional[LLM]): Language model to use.

    Examples:
        `pip install llama-index-experimental`

        ```python
        import pandas as pd
        from llama_index.experimental.query_engine.pandas import PandasQueryEngine

        df = pd.DataFrame(
            {
                "city": ["Toronto", "Tokyo", "Berlin"],
                "population": [2930000, 13960000, 3645000]
            }
        )

        query_engine = PandasQueryEngine(df=df, verbose=True)

        response = query_engine.query("What is the population of Tokyo?")
        ```

    """

    def __init__(
        self,
        df: pd.DataFrame,
        instruction_str: Optional[str] = None,
        instruction_parser: Optional[PandasInstructionParser] = None,
        pandas_prompt: Optional[BasePromptTemplate] = None,
        output_kwargs: Optional[dict] = None,
        head: int = 5,
        verbose: bool = False,
        service_context: Optional[ServiceContext] = None,
        llm: Optional[LLM] = None,
        synthesize_response: bool = False,
        response_synthesis_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._df = df

        self._head = head
        self._pandas_prompt = pandas_prompt or DEFAULT_PANDAS_PROMPT
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._instruction_parser = instruction_parser or PandasInstructionParser(
            df, output_kwargs or {}
        )
        self._verbose = verbose

        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._synthesize_response = synthesize_response
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )

        super().__init__(
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            )
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "pandas_prompt": self._pandas_prompt,
            "response_synthesis_prompt": self._response_synthesis_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "pandas_prompt" in prompts:
            self._pandas_prompt = prompts["pandas_prompt"]
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    @classmethod
    def from_index(cls, index: PandasIndex, **kwargs: Any) -> "PandasQueryEngine":
        logger.warning(
            "PandasIndex is deprecated. "
            "Directly construct PandasQueryEngine with df instead."
        )
        return cls(df=index.df, service_context=index.service_context, **kwargs)

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self._df.head(self._head))

    def _query(self, query_bundle: QueryBundle):
        """Answer a query."""
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )
        # print(pandas_response_str)
        pandas_response_str = "df"
        if self._verbose:
            print_text(f"> Pandas Instructions:\n" f"```\n{pandas_response_str}\n```\n")
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        
        if self._verbose:
            print_text(f"> Pandas Output: {pandas_output}\n")

        response_metadata = {
            "pandas_instruction_str": pandas_response_str,
            "raw_pandas_output": pandas_output,
        }

        if self._synthesize_response:
            response_gen = self._llm.stream(
                    self._response_synthesis_prompt,
                    query_str=query_bundle.query_str,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
            )
            return StreamingResponse(response_gen=response_gen)
        else:
            response_str = str(
                self._llm.predict(
                    self._response_synthesis_prompt,
                    query_str=query_bundle.query_str,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
            response_str = str(pandas_output)

            return Response(response=response_str, metadata=response_metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)


# legacy
NLPandasQueryEngine = PandasQueryEngine
GPTNLPandasQueryEngine = PandasQueryEngine
