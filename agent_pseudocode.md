```python
TOOLS = [TOOL_1, TOOL_2]

TASK = """
I'm hungry and I want some fruit. Please help me figure out how much it will 
cost...possibly using the tools I've given you!
"""

conversation_messages = assemble_initial_conversation_messages(SYSTEM_PROMPT, TASK)

MAX_STEPS = 20
response = call_model_with_tools(
    model="gpt-5", input=conversation_messages, tools=TOOLS, store=True
)
for step in range(MAX_STEPS):

    if not has_tool_calls(response):
        break

    tool_call_object = get_tool_call(response)
    tool_call_result = tool_call_runner(tool_call_object)

    response = call_model_with_tools(
        model="gpt-5",
        previous_response_id=response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call_object.call_id,
                "output": json.dumps(tool_call_result),
            }
        ],
        tools=TOOLS,
    )

    num_steps += 1
    if num_steps >= MAX_STEPS:
        break

# optional summary
call_model_without_tools(
    model="gpt-5",
    previous_response_id=response.id,
    input=conversation_messages
    + [
        {
            "role": "user",
            "content": "Can you please summarize the results of the tool calls you just "
            "made? What changed in the repo?",
        }
    ],
)
```