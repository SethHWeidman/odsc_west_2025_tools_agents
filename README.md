# ODSC West 2025 talk on tools and agents

## Steps to run "GPTree" demo

1. `cd` into https://github.com/SethHWeidman/local-gpt
2. Copy over the [`01_chat_completions_demo.py`](01_chat_completions_demo.py) script
3. Run `python 01_chat_completions_demo.py`

## Script descriptions

The [`01_chat_completions_demo.py`](01_chat_completions_demo.py),
[`02_responses_api_demo.py`](02_responses_api_demo.py), and
[`03_responses_api_stateful_demo.py`](03_responses_api_stateful_demo.py) scripts all
illustrate making a simple code change to a repo using functionality from the OpenAI
APIs.

* [`01_chat_completions_demo.py`](01_chat_completions_demo.py) uses the Chat Completions
  API
* [`02_responses_api_demo.py`](02_responses_api_demo.py) uses the Responses API
  (stateless)
* [`03_responses_api_stateful_demo.py`](03_responses_api_stateful_demo.py) uses the
  Responses API with `store=True` to make the model save the conversation state

See
[here](https://platform.openai.com/docs/guides/migrate-to-responses#3-update-multi-turn-conversations)
for the distinction between how the latter two methods are used in the Responses API.