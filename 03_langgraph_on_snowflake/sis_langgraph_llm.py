import streamlit as st
from typing import Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain import SQLDatabase
from langchain.llms.base import LLM
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
st.header("LangGraph")
llm = SnowflakeLLM(session=session, model="claude-3-5-sonnet")


# 状態の定義
class State(TypedDict):
    user_input: str
    llm_response: str


# ノード関数の定義
def process_user_input(state: State) -> State:
    """ユーザー入力を処理し、LLM に渡す"""
    user_input = state["user_input"]
    # 必要に応じて前処理を行う
    processed_input = user_input.strip()
    state["user_input"] = processed_input
    return state

def generate_llm_response(state: State) -> State:
    """LLM を使用して応答を生成する"""
    user_input = state["user_input"]
    response = llm.invoke(user_input)
    state["llm_response"] = response
    return state

def display_response(state: State) -> State:
    """LLM の応答を表示する"""
    print("LLM Response:", state["llm_response"])
    return state

# StateGraph の初期化
graph_builder = StateGraph(State)

# ノードの追加
graph_builder.add_node("process_user_input", process_user_input)
graph_builder.add_node("generate_llm_response", generate_llm_response)
graph_builder.add_node("display_response", display_response)

# エッジの定義
graph_builder.add_edge(START, "process_user_input")
graph_builder.add_edge("process_user_input", "generate_llm_response")
graph_builder.add_edge("generate_llm_response", "display_response")
graph_builder.add_edge("display_response", END)

# グラフのコンパイル
compiled_graph = graph_builder.compile()

# 初期状態の定義
initial_state = {"user_input": "こんにちは、今日の天気は？", "llm_response": ""}

# ワークフローの実行
compiled_graph.invoke(initial_state)
