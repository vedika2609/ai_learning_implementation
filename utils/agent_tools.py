from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI, responses
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.types import Command
from langgraph.graph import StateGraph
from typing import Literal, Any
from dotenv import load_dotenv


dotenv.load_dotenv()

MODEL = "gpt-4.1-nano"
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
State = Dict[str, Any]

def openai_sdk_demo():
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Summarize LangGraph in one line."}]
    )
    print(response.choices[0].message.content)

def langchain_demo():
    llm = ChatOpenAI(model=MODEL)
    prompt = ChatPromptTemplate.from_template("Summarize this text:\n{input}")
    chain = prompt | llm
    print(chain.invoke({"input: LangGraph is a grpah based orchestration framework."}))

def langgraph_demo():
    # LangGraph is a graph-based orchestrator designed for multi-agent systems.
    """
    each node can :
        run a LLM
        update shared state
        tell LangGraph what node to run next using `Command`
    """

def define_llm_node():
    llm = ChatOpenAI(model=MODEL, temperature=0.0)

def call_llm_node(state: State, config=None, runtime=None) -> State:
    prompt = state.get("user_prompt", "Say hello")
    messages = [
        SystemMessage(content="You are a helpful assistant that answers YES or NO clearly when asked if the answer exists."),
        HumanMessage(content=f"Question: {prompt}\n\nIf you can directly answer the question, reply with 'YES' and then the short answer. If not, reply 'NO' and explain what you need.")
    ]



if __name__ == '__main__':
    # openai_sdk_demo()
    langgraph_demo()
