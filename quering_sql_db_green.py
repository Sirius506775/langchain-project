
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
import os

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

db = SQLDatabase.from_uri(os.environ['DATABASE_URL'])

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

print(sql_response.invoke({"question": "명륜3가인 데이터가 뭐야?"}))

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

print(full_chain.invoke({"question": "크기가 제일 큰 데이터의 lat, lon, house_Type을 알려줘. 답변은 한글로 출력해줘"}).content)
# print(full_chain.invoke({"question": "Canada에서 살고 있는 employee 중에 email이 andrew@chinookcorp.com인 employee의 birthdate는 무엇인가요?"}).content)
# """