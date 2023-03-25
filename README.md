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
**Creating a Conversation Agent optimized for chatting**:
*codesnippet:
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
