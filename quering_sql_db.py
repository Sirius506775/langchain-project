
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase


template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

# sqlite3 Chinook.db
# .read Chinook_Sqlite.sql
# download the sql file from the link below
# https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql

db = SQLDatabase.from_uri("sqlite:///./Chinook.db")

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

model = ChatOpenAI()

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

print(sql_response.invoke({"question": "Canada에서 살고 있는 employee 중에 email이 andrew@chinookcorp.com인 employee의 birthdate는 무엇인가요?"}))
"""
SELECT COUNT(*) FROM Employee
"""

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: run_query(x["query"]),
    )
    | prompt_response
    | model
)

print(full_chain.invoke({"question": "Canada에서 살고 있는 employee 중에 email이 andrew@chinookcorp.com인 employee의 birthdate는 무엇인가요?"}))
print(full_chain.invoke({"question": "Canada에서 살고 있는 employee 중에 email이 andrew@chinookcorp.com인 employee의 birthdate는 무엇인가요?"}).content)
"""
content='There are 8 employees.'
"""