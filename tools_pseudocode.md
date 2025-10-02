```python
# Suppose TOOL_1 and TOOL_2 are defined in Python as
# 
# def tool_1(meat: int, cheese: int) -> float:
#     """Get the total price of some meat and cheese."""
#     return 5.0 * meat + 6.5 * cheese

# and TOOL_2 was defined as 

# def tool_2(apples: int, bananas: int) -> float:
#     """Get the total price of some fruit."""
#     return 3.4 * apples + 1.7 * bananas
#
# (the actual function definitions would be a JSON version of this)


TOOLS = [TOOL_1, TOOL_2]
# shortly, we will define TOOLS = [BASH_TOOL]

TASK = """
I'm hungry and I want some fruit. Please help me figure out how much it will 
cost...possibly using the tools I've given you!
"""

initial_conversation_messages = assemble_initial_conversation_messages(
    SYSTEM_PROMPT, TASK
)

response = call_model_with_tools(
    model="gpt-5",
    input=initial_conversation_messages,
    tools=TOOLS,
    # with OpenAI's APIs, we could optionally *require* the model to call tools; we will
    # do this shortly
)

# response has some structure like:
# {
#     "output": [
#         {
#             "type": "reasoning",
#             "text": "I should use these handy tools my maker has given me!"
#         },
#         {
#             "type": "function_call",
#             "id": "123"
#             "name": "tool_2"
#             "arguments": {'bananas': 3, 'apples': 5}
#         },
#     }
# }

tool_call = {
    "type": "function_call",
    "id": "123",
    "name": "tool_2",
    "arguments": {'bananas': 3, 'apples': 5}
}
# We would need to define some "tool_call_runner"
# 
# def tool_call_runner(function_call: openai.ToolCallObject) -> typing.Any:
#     """
#.    Takes in ToolCallObject results like:
# 
#         {
#             "type": "function_call",
#             "id": "123"
#             "name": "tool_2"
#             "arguments": {'bananas': 3, 'apples': 5}
#         }
#
#     converts them to Python functions, and runs them.
#     """
# 

tool_call_result = tool_call_runner(tool_call)

summary = summarize_final_result(
    model="gpt-5",
    previous_response_id=resp.id,
    input=tool_call_result
    + {
        "role": "user",
        "content": "Can you please summarize the results of the tool call you just "
        "made? What changed in the repo?",
    },
)
print(summary)
```