import json
import os
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.sql_database import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine


def save_chat_to_file(session_id, messages):
    with open(f"chat_history_{session_id}.json", "w") as f:
        json.dump(messages, f)

def load_chat_from_file(session_id):
    try:
        with open(f"chat_history_{session_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

st.set_page_config(page_title="Langchain: Chat with SQL DB")
st.title("Langchain : Chat with SQL DB")

localDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["Use SQLite3 database - student.db", "Connect to your SQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB to chat with", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = localDB

api_key = st.sidebar.text_input(label="Provide Secret key", type="password")

if api_key!="123Ani":
    st.info("Please add Secret Key")
    st.stop()


################################ Open Source LLM
# llm = ChatGroq(model_name="Llama3-8b-8192", api_key=api_key, streaming=True)

################################ Azure ChatGPT
# os.environ['OPENAI_API_VERSION'] = os.getenv("AZURE_OPENAI_API_VERSION")
# os.environ['OPENAI_CHATGPT_DEPLOYMENT'] = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
# os.environ['OPENAI_CHATGPT_MODEL'] = os.getenv("AZURE_OPENAI_CHATGPT_MODEL")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv("AZURE_OPENAI_RESOURCE")

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("AZURE_OPENAI_CHATGPT_MODEL")
azure_endpoint = os.getenv("AZURE_OPENAI_RESOURCE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the LangChain ChatOpenAI client for Azure
llm = AzureChatOpenAI(
    deployment_name="Test-1-0",  # Your deployment name
    model_name=model_name,   # Or whatever model you've deployed
    openai_api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,  # Use your actual version
    temperature=0.7
)
################################ Configuring DB ###################
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == localDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite://", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_db and mysql_password):
            st.error("Please provide all MySQL connection details")
            st.stop()
        return SQLDatabase(create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        ))

# Init DB
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

##################################### SQL Toolkit and Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)
################################### Routing Query Manually 
# Prompt to classify user input
chat_template = PromptTemplate.from_template("You are a friendly assistant. Reply to: {input}")
chat_chain = LLMChain(llm=llm, prompt=chat_template)

classifier_prompt = PromptTemplate.from_template(
    """You are a classification assistant. Classify the following user query strictly as one of these two categories only: SQL or GENERAL.
Return only the word "SQL" or "GENERAL" as your response.

Query: {input}
Classification:"""
)
classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt)

def route_query(query: str):
    result = classifier_chain.invoke({"input": query})
    classification = result.get("text", "").strip().lower()
    print("classification - ", type(classification), "\n", classification)
    if classification == "sql":
        return "sql"
    return "general"

########################## History wrapper
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

########################## Trimming history ###############################
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=llm.get_num_tokens_from_messages,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
################## Implementing history ###################
def get_session_history(session_id):
    history = ChatMessageHistory()
    messages = st.session_state.get("messages", [])[-10:]
    for m in messages:
        if m["role"] == "user":
            content = m["content"]["input"] if isinstance(m["content"], dict) else m["content"]
            history.add_user_message(content)
        else:
            content = (
                m["content"].get("output") if isinstance(m["content"], dict) and "output" in m["content"]
                else m["content"]
            )
            if content is not None:
                history.add_ai_message(content)
    history.messages = trimmer.invoke(history.messages)
    return history

# agent_with_history = RunnableWithMessageHistory(
#     router_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history"
# )
sql_agent_with_history = RunnableWithMessageHistory(
    sql_agent,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

chat_chain_with_history = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
########################################## UI State
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
    

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database.")
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        route = route_query(user_query)
        if route == "sql":
            response = sql_agent_with_history.invoke(
                {"input": user_query},
                config={
                    "callbacks": [streamlit_callback],
                    "configurable": {"session_id": st.session_state.session_id}
                }
            )
            assistant_reply = response.get("output") if isinstance(response, dict) else str(response)
        else:
            response = chat_chain_with_history.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            assistant_reply = response.get("text") if isinstance(response, dict) else str(response)
        # response = agent_with_history.invoke(
        #     {"input": user_query},
        #     config={"configurable": {"session_id": st.session_state.session_id}},
        #     callbacks=[streamlit_callback]
        # )
        # st.session_state["messages"].append({"role": "assistant", "content": response})
        # st.write(response)
        print(response)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
        st.write(assistant_reply)
        save_chat_to_file(st.session_state.session_id, st.session_state["messages"])


