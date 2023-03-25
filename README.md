# LangChain-CheatSheet
LangChain CheatSheet


LANGCHAIN Async API for Agent
Cheat Sheet:

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
