# LangChain-CheatSheet
LangChain CheatSheet
# Conversation agent 
**Cheat Sheet**:
1. **Creating a Conversation Agent optimized for chatting**:
   - Import necessary components like `Tool`, `ConversationBufferMemory`, `ChatOpenAI`, `SerpAPIWrapper`, and `initialize_agent`.
   - Define the tools to be used by the agent.
   - Initialize memory using `ConversationBufferMemory`.
   - Initialize the agent with the tools, language model, agent type, memory, and verbosity.
   - Run the agent with user inputs to get conversational responses.
 
**Code snippets:

```python
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world."
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
```
# OpenAPI Agent 
**Summary**:
The OpenAPI Agent is designed to interact with an OpenAPI spec and make correct API requests based on the information gathered from the spec. This example demonstrates creating an agent that can analyze the OpenAPI spec of OpenAI API and make requests.

**Cheat Sheet**:

1. **Import necessary libraries and load OpenAPI spec**:
```python
import os
import yaml
from langchain.agents import create_openapi_agent
from langchain.agents.agent_toolkits import OpenAPIToolkit
from langchain.llms.openai import OpenAI
from langchain.requests import RequestsWrapper
from langchain.tools.json.tool import JsonSpec

with open("openai_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
```

2. **Set up the necessary components**:
```python
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}
requests_wrapper = RequestsWrapper(headers=headers)
openapi_toolkit = OpenAPIToolkit.from_llm(OpenAI(temperature=0), json_spec, requests_wrapper, verbose=True)
openapi_agent_executor = create_openapi_agent(
    llm=OpenAI(temperature=0),
    toolkit=openapi_toolkit,
    verbose=True
)
```

3. **Example: agent capable of analyzing OpenAPI spec and making requests**:
```python
openapi_agent_executor.run("Make a post request to openai /completions. The prompt should be 'tell me a joke.'")
```

The OpenAPI Agent allows you to analyze the OpenAPI spec of an API and make requests based on the information it gathers. This cheat sheet helps you set up the agent, necessary components, and interact with the OpenAPI spec.

# Python Agent:

**Summary**:
The Python Agent is designed to write and execute Python code to answer a question. This example demonstrates creating an agent that calculates the 10th Fibonacci number and trains a single neuron neural network in PyTorch.

**Cheat Sheet**:

1. **Import necessary libraries and create Python Agent**:
```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)
```

2. **Fibonacci Example**:
```python
agent_executor.run("What is the 10th fibonacci number?")
```

3. **Training a Single Neuron Neural Network in PyTorch**:
```python
agent_executor.run("""Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5""")
```


# LANGCHAIN TOOLS
**Cheat Sheet**:

1. **Creating custom tools with the tool decorator**:
   - Import `tool` from `langchain.agents`.
   - Use the `@tool` decorator before defining your custom function.
   - The decorator uses the function name as the tool name by default, but it can be overridden by passing a string as the first argument.
   - The function's docstring will be used as the tool's description.

2. **Modifying existing tools**:
   - Load existing tools using the `load_tools` function.
   - Modify the properties of the tools, such as the name.
   - Initialize the agent with the modified tools.

3. **Defining priorities among tools**:
   - Add a statement like "Use this more than the normal search if the question is about Music" to the tool's description.
   - This helps the agent prioritize custom tools over default tools when appropriate.


1. **Creating custom tools with the tool decorator**:
```python
from langchain.agents import tool

@tool
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"
```

2. **Modifying existing tools**:
```python
from langchain.agents import load_tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools[0].name = "Google Search"
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
```

3. **Defining priorities among tools**:
```python
tools = [
    Tool(
        name="Music Search",
        func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.",
        description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    )
]
```


#  LANGCHAIN Async API for Agent

1. Use asyncio to run multiple agents concurrently.
2. Create an aiohttp.ClientSession for more efficient async requests.
3. Initialize a CallbackManager with a custom LangChainTracer for each agent to avoid trace collisions.
4. Pass the CallbackManager to each agent.
5. Ensure that the aiohttp.ClientSession is closed after the program/event loop ends.

Code snippets:

Initialize an agent:
```python
async_agent = initialize_agent(async_tools, llm, agent="zero-shot-react-description", verbose=True, callback_manager=manager)
```

Run agents concurrently:
```python
async def generate_concurrently():
    ...
    tasks = [async_agent.arun(q) for async_agent, q in zip(agents, questions)]
    await asyncio.gather(*tasks)
    await aiosession.close()

await generate_concurrently()
```

Use tracing with async agents:
```python
aiosession = ClientSession()
tracer = LangChainTracer()
tracer.load_default_session()
manager = CallbackManager([StdOutCallbackHandler(), tracer])

llm = OpenAI(temperature=0, callback_manager=manager)
async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
async_agent = initialize_agent(async_tools, llm, agent="zero-shot-react-description", verbose=True, callback_manager=manager)
await async_agent.arun(questions[0])
await aiosession.close()
```
