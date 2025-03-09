# Pythonは3.11を使用しました。
from typing import Any, Optional

import streamlit as st
from langchain import SQLDatabase
from langchain.llms.base import LLM
from langchain_snowflake import CortexSearchRetriever
from snowflake.cortex import complete
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session


class SnowflakeLLM(LLM):
    model: str = "claude-3-5-sonnet"
    session: Session = None
    options: Optional[dict[str, Any]] = None

    @property
    def _llm_type(self) -> str:
        return "snowflake"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "session": self.session}

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None
    ) -> str:
        st.write(prompt)
        response = complete(
            model=self.model,
            prompt=prompt,
            options=self.options,
            session=self.session,
        )
        st.write(response)
        if stop:
            indexes = [response.find(token) for token in stop if token in response]
            if indexes:
                min_index = min(indexes)
                response = response[:min_index]
        
        return response

session = get_active_session()

## Complete
st.header("Complete")
snow_llm = SnowflakeLLM(session=session, model="claude-3-5-sonnet")
st.write(snow_llm("こんにちは"))

_="""
user_input = st.text_input("プロンプトを入力してください", "SnowflakeのLLM機能について教えてください。")
if user_input:
    result = snow_llm(user_input)
    st.write("【応答】")
    st.write(result)
"""

## Cortex Search
st.header("Cortex Search")
from langchain.tools import BaseTool
from pydantic import BaseModel
from typing import ClassVar, Type, Optional

class SearchInput(BaseModel):
    query: str

class CortexSearchRetrieverTool(CortexSearchRetriever, BaseTool):
    name: ClassVar[str] = "cortex_search"
    description: ClassVar[str] = "Cortex Searchツール。Cortexを利用してRAG検索を実行します。"
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, query: str, run_manager: Optional[object] = None
    ) -> str:
        st.write(query)
        response = self.invoke(query)
        st.write(response)
        return response
        
snow_search = CortexSearchRetrieverTool(
    sp_session=session,
    database="sales_intelligence",
    schema="data",
    search_service="sales_conversation_search",
    search_column="transcript_text",
    columns=["transcript_text", "customer_name"],
    limit=3
)


## Snowflake Database
st.header("Database")
from pydantic import BaseModel
from langchain.tools import BaseTool
from typing import Literal, Union
import pandas as pd

from snowflake.snowpark.session import Session
import snowflake.snowpark.types as T
import snowflake.snowpark.functions as F
from snowflake.snowpark import DataFrame as SnowparkDataFrame

# Snowparkとの連携を行うクラス
class SnowparkSQLAdapter:
    """Snowparkと通信するコアクラス。ツールクラスはこのクラスを使用してSnowflakeと通信します。"""

    def __init__(self, session: Session, database: str = None, schema: str = None):
        """
        session : Snowparkセッション。Snowflakeとのやり取りに使用します。
        """
        self._sp_session = session
        self.database = database
        self.schema = schema

    def _execute(self, sql_stmt: str) -> SnowparkDataFrame:
        spdf = self._sp_session.sql(sql_stmt)
        return spdf

    def run(self, p_sql_stmt: str, fetch: Union[Literal["all"], Literal["one"]] = "all") -> SnowparkDataFrame:
        sql_stmt = p_sql_stmt.replace(';', '')
        spdf = self._execute(sql_stmt)
        if fetch == 'one':
            return spdf.limit(1)
        # デフォルトでは全ての行を返す
        return spdf

# ツールクラスのベースとなるクラス
class SnowparkSQLInput(BaseModel):
    snowpark_adapter: SnowparkSQLAdapter

    model_config = {
        'arbitrary_types_allowed': True
    }

# クエリ実行用のツールクラス
class SnowparkQueryTool(SnowparkSQLInput, BaseTool):
    name: str = 'snowpark_sql'
    description: str = '''Snowflake内のデータをクエリし操作するためのツール。
    入力は詳細かつ正確なSnowflake SQLクエリで、出力はpandas DataFrame形式の結果です。
    ステートメントが行を返す場合、pandas DataFrameが返されます。
    ステートメントが行を返さない場合、空のpandas DataFrameが返されます。'''

    def _run(self, sql_query: str) -> pd.DataFrame:
        st.write(sql_query)
        try:
            spdf = self.snowpark_adapter.run(sql_query)
            if "select" in sql_query.lower():
                df = spdf.toPandas()
            else:
                import pandas as pd
                df = pd.DataFrame(spdf.collect())    
            return df
        except Exception as e:
            return f"クエリ（{sql_query}）の実行中に次のエラーが発生しました：{e}"

# テーブル一覧取得用のツールクラス
class SnowparkListTablesTool(SnowparkSQLInput, BaseTool):
    name: str = 'snowpark_tables'
    description: str = '''Snowflakeの現在のSnowparkセッションでアクセス可能なテーブルの一覧を取得するツール。'''

    def _run(self, dummy_input: str = 'dummy') -> pd.DataFrame:
        if self.snowpark_adapter.database:
            db = self.snowpark_adapter.database
        else:
            db = self.snowpark_adapter._sp_session.get_current_database()
        sql_stmt = f'''
            SELECT table_catalog, table_schema, table_name, table_type, comment
            FROM {db}.INFORMATION_SCHEMA.TABLES
            WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
        '''
        spdf = self.snowpark_adapter.run(sql_stmt)
        df = spdf.toPandas()
        return df


from langchain.agents import initialize_agent, load_tools, Tool
_= """
adapter = SnowparkSQLAdapter(session, database="snowflake_sample_data", schema="public")
list_tables_tool = SnowparkListTablesTool(snowpark_adapter = adapter)
query_tool = SnowparkQueryTool(snowpark_adapter = adapter)

st.write(list_tables_tool.name)
st.write(list_tables_tool.description)
df = list_tables_tool.run({})
st.dataframe(df)

tools = [
    SnowparkQueryTool(snowpark_adapter = adapter),
    SnowparkListTablesTool(snowpark_adapter = adapter),
]

agent = initialize_agent(tools, snow_llm, agent="zero-shot-react-description", verbose=True)
response = agent.run('TPCH_SF1のCUSTOMERテーブルとORDERSテーブルなどを使って行える分析のサンプルを示してください。クエリを作成する際は、完全修飾子を使うようにしてください。')
st.write(response)
"""

adapter = SnowparkSQLAdapter(session, database="sales_intelligence", schema="data")
list_tables_tool = SnowparkListTablesTool(snowpark_adapter = adapter)
query_tool = SnowparkQueryTool(snowpark_adapter = adapter)

st.write(list_tables_tool.name)
st.write(list_tables_tool.description)
df = list_tables_tool.run({})
st.dataframe(df)

tools = [
    SnowparkQueryTool(snowpark_adapter = adapter),
    SnowparkListTablesTool(snowpark_adapter = adapter),
    snow_search
]

agent = initialize_agent(tools, snow_llm, 
                         agent="zero-shot-react-description", verbose=True)

if st.button("実行する"):
    response = agent.run("""RAGやDBを使って、売上No1の営業担当の会話を分析してください。
                            クエリを作成する際は、完全修飾子を使うようにして、
                            かつテーブルの情報を調べるようにしてください。""")
    st.write(response)
