# Simple ReAct Agent from Scratch

This project is a "from scratch" implementation of a basic AI agent using the **ReAct (Reason+Act)** pattern. It demonstrates the foundational principles of agentic AI—state management, tool use, and orchestration—using only standard Python libraries and an OpenAI API connection. It serves as an educational tool to understand agent mechanics before moving to advanced frameworks like LangGraph.



---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository
First, clone this repository to your local machine using Git.

```bash
git clone [https://github.com/devtraldi/AI_Agents_in_LangGraph.git](https://github.com/devtraldi/AI_Agents_in_LangGraph.git)
cd AI_Agents_in_LangGraph
````

### 2\. Create and Activate a Virtual Environment

It is best practice to use a virtual environment to manage project-specific dependencies.

**On macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3\. Install Dependencies

Install the required Python libraries using pip. A `requirements.txt` file is included for convenience.

```bash
pip install -r requirements.txt
```

This will install key libraries including `openai` and `python-dotenv`.

### 4\. Set Up Your API Key

The agent requires an OpenAI API key to function. This key should be stored securely in an environment file.

1.  Create a new file in the root of the project directory named `.env`.

2.  Add your API key to this file in the following format:

    ```
    OPENAI_API_KEY="sk-YourSecretApiKeyGoesHere"
    ```

**Important:** The `.env` file is listed in `.gitignore` and should **never** be committed to version control.

-----

## ⚡️ Running the Agent

Once the setup is complete, you can run the agent by executing the main Python script (`main.py`) or the Jupyter Notebook (`agent_notebook.ipynb`).

-----

## 🧠 How It Works: A Deep Dive

This section breaks down the code and concepts behind the agent, framed as an explanation for a technical interview.

### The High-Level Overview

This script is a foundational implementation of an AI agent built from scratch using the **ReAct pattern**, which stands for **Reason** and **Act**. The goal was to build a deep, fundamental understanding of how agents work before leveraging higher-level frameworks like LangChain or LangGraph.

The core idea of ReAct is to augment a Large Language Model (LLM) with the ability to use external **tools**. By itself, an LLM can't perform real-time calculations or look up specific, non-public data. This agent solves that by reasoning about a problem, choosing a tool, executing it, and observing the result to inform its next step. This implementation demonstrates the three essential components of any agent: **state management**, **tool invocation**, and **orchestration logic**.

### Block 1: Imports and API Configuration

First, we begin with the initial setup.

```python
import openai
import re
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI
client = OpenAI()
```

Here, we import the necessary libraries: `openai` is the client library to interact with the API, `re` is for regular expressions to parse the model's output, and `os` and `dotenv` are used for securely managing our API key.

The line `_ = load_dotenv()` executes the function to load environment variables from a `.env` file in the project's root directory. We then instantiate the `OpenAI` client, which automatically authenticates using the `OPENAI_API_KEY` we just loaded. This `client` object is our gateway to the LLM.

### Block 2: Tool Definition

Next, we define the set of tools our agent is capable of using.

```python
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    # ... function logic ...

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
```

These are simple Python functions that perform specific, well-defined tasks. The `calculate` function takes a string expression and uses Python's `eval` to compute the result. The `average_dog_weight` function acts as a mock database or a simple knowledge base lookup.

The `known_actions` dictionary serves as a **tool registry** or a **dispatch table**. It maps the string name of an action, which the LLM will generate, to the actual callable Python function. This is how our orchestration logic will know which function to execute based on the model's output.

### Block 3: The System Prompt

This multi-line string is the agent's **constitution** or its core operating instructions. It's the most critical piece for guiding the model's behavior.

```python
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
...
""".strip()
```

We instruct the model to follow a strict cycle:

  * **Thought:** The model first verbalizes its reasoning process.
  * **Action:** Based on its thought, it must output a specific, parsable line, like `Action: calculate: 37 + 20`. This is the command for our code to execute. The `PAUSE` token signals that it's waiting for an external result.
  * **Observation:** Our code will then feed the result of the action back to the model in a new prompt, prefixed with `Observation:`.

By enforcing this structured format, we make the LLM's output predictable and machine-readable, which is essential for the automation loop.

### Block 4: The `Agent` Class

The `Agent` class encapsulates the agent's core logic, managing its memory and its interaction with the LLM.

```python
class Agent:
    def __init__(self, system=""):
        # ...
    def __call__(self, message):
        # ...
    def execute(self):
        # ...
```

  * The **`__init__` method** is the constructor. It initializes the agent's state. It takes the `system` prompt and creates an empty list called `self.messages`. This list will serve as the agent's **memory**, storing the entire history of the conversation.

  * The **`__call__` method** makes the class instance callable, like a function. This is the main entry point for sending a new message to the agent. It appends the new user message to the history, calls `self.execute()` to get a response from the LLM, and then appends the LLM's response to the history, thus maintaining the conversational state.

  * The **`execute` method** handles the direct communication with the OpenAI API. It bundles the entire `self.messages` history and sends it to the specified model. We set `temperature=0` to ensure the output is as deterministic and predictable as possible, which is crucial for a ReAct loop. It then parses the response and returns the LLM's raw text output.

### Block 5: The Orchestration Loop

Finally, if the `Agent` class is the brain, the `query` function is the **orchestrator** or the central nervous system. It drives the entire ReAct process.

```python
action_re = re.compile(r'^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    # ... loop logic ...
```

First, we define `action_re`, a regular expression used to find and parse the `Action:` line from the model's output.

The `query` function works as follows:

1.  It initializes a new `Agent` instance.
2.  It enters a `while` loop that will run for a maximum of `max_turns` to prevent infinite loops.
3.  **Reason:** Inside the loop, it calls the agent with the current prompt (`next_prompt`), which returns the model's `Thought` and `Action`.
4.  **Act:** It then uses our regular expression to check if the output contains a valid `Action`.
5.  If an action is found, it extracts the action name (e.g., `calculate`) and its input (e.g., `37 + 20`). It looks up the function in our `known_actions` dictionary and executes it.
6.  **Observe:** The result of that function call is then formatted into an `Observation:` string. This string becomes the `next_prompt` for the next iteration of the loop.
7.  The loop continues until the model outputs a final `Answer:` instead of an `Action:`, at which point the function returns.

-----

## Conclusion

In summary, this script builds a complete agent from first principles. It clearly separates the LLM's reasoning capabilities from the deterministic execution of tools. By managing a conversational memory and using an orchestration loop to interpret and act on the model's structured output, we create a system that can solve multi-step problems that are beyond the scope of a simple LLM call. This foundational knowledge is directly applicable when using advanced frameworks, which essentially manage these same components at a higher level of abstraction.

```
```