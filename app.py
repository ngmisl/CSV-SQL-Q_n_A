import pandas as pd
import sqlite3
import streamlit as st
import base64
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    DateTime,
)
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
import os
from streamlit_chat import message


# Infer column types from CSV
def infer_column_types(df):
    type_map = {
        "int64": Integer,
        "float64": Float,
        "datetime64": DateTime,
        "object": String,
    }
    return [type_map[str(df[col].dtype)] for col in df.columns]


# Import CSV data into the database
def csv_to_sqlite(df, db_file, table_name):
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def get_binary_file_downloader_html(bin_file, file_label="File"):
    with open(bin_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href


st.set_page_config(
    page_title="CSV to SQLite",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("CSV to SQLite Converter")

# Add OpenAI API Key input field and CSV file uploader to the sidebar
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your API key:", type="password")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Add a Follow Us widget to the sidebar
st.sidebar.title("Follow Us")
st.sidebar.markdown(
    """
    * [Web](https://ngmi.ai/)
    * [Mastodon](https://mastodon.online/@ngmi)
    * [GitHub](https://github.com/ngmisl)
    """
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

if uploaded_file and api_key:
    # Read CSV file using pandas
    df = pd.read_csv(uploaded_file)

    # Get input file name and replace the extension with .sqlite
    db_file = uploaded_file.name.replace(".csv", ".sqlite")
    table_name = "input_table"

    # Create database schema
    engine = create_engine(f"sqlite:///{db_file}")
    metadata_obj = MetaData()

    column_types = infer_column_types(df)

    wallet_stats_table = Table(
        table_name,
        metadata_obj,
        *[
            Column(column_name, column_type)
            for column_name, column_type in zip(df.columns, column_types)
        ],
    )

    metadata_obj.create_all(engine)

    csv_to_sqlite(df, db_file, table_name)

    # Set up the SQL agent
    db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
    llm = OpenAI(client=None, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

    # Initialize the chat history in the session_state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Enter your question:", key="input_field")

    if user_input:
        answer = agent_executor.run(user_input)
        # Add the question and answer to the chat_history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("agent", answer))

    # Display the chat_history in a chat-like format using streamlit-chat
    for i, (sender, message_text) in enumerate(st.session_state.chat_history):
        if sender == "user":
            message(message_text, is_user=True, key=f"{i}_user")
        else:
            message(message_text, key=f"{i}")

    st.success(f"CSV data has been imported into {db_file}.")
    st.markdown(
        get_binary_file_downloader_html(db_file, "Download SQLite Database"),
        unsafe_allow_html=True,
    )
