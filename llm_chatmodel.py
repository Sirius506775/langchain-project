from langchain_openai import OpenAI, ChatOpenAI

# Configure OpenAI API key
# export OPENAI_API_KEY=""

llm = OpenAI()
chat_model = ChatOpenAI()

from langchain_core.messages import HumanMessage, SystemMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [SystemMessage(content="You are the teenager"),
            HumanMessage(content=text)]

# Takes in a string, returns a string.
print(llm.invoke(text))

# Takes in a list of BaseMessage, returns a BaseMessage.
print(chat_model.invoke(messages))