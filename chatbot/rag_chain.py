import os
import json
from dotenv import load_dotenv
from groq import Groq

# --- TOOL FUNCTIONS ---
from chatbot.weatherTool import get_weather
from chatbot.locationTool import get_location

# --- Load API key and instantiate Groq client ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"  # Confirm your Groq model list

# --- TOOL DEFINITIONS ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location to fetch weather for."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "Get the user's current location based on IP address. Use for questions like 'my location', 'where am I', etc.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# --- Function dispatcher for tool calls ---
def function_router(function_name, arguments):
    if function_name == "get_weather":
        return get_weather(**arguments)
    elif function_name == "get_location":
        return get_location()
    else:
        return "Tool not implemented."

# --- Retrieve RAG context using vectorstore as argument ---
def retrieve_context(question: str, vectorstore, k: int = 5) -> str:
    """
    Retrieve relevant documents from vectorstore for a given question.
    Returns the concatenated context string.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])

# --- Main callable: to be used in app.py ---
def answer_with_rag_and_tools(question, vectorstore, history=None):
    """
    RAG+Tools Groq orchestration.
    arguments:
        question: str, user input
        vectorstore: your vectorstore instance (must support as_retriever().invoke)
        history: list of dicts (optional), prior messages for chat context
    returns:
        str, final assistant answer
    """
    if history is None:
        history = []

    # Compose messages for the LLM
    messages = [
        {
            "role": "system",
            "content": (
              "You are a helpful assistant. Use the provided context to answer questions. "
            "If (and only if) the user asks about weather, call ONLY the get_weather tool. "
"If (and only if) the user asks about their location, call ONLY the get_location tool. "
"If the user asks about both weather AND location in the same question, call both tools."
"Do NOT call a tool unless the user's question is explicitly about weather or location."
"Never call get_weather unless the user asks for weather or temperature explicitly and provides a city or location; never call get_weather with an empty or unknown location."
"Combine only that specific  tool (get_location tool or get_weather tool depending upon user query) and context outputs if needed to answer all parts of the user's request, clearly and completely."
"If user asks about weather , location and context ,call both tools and context output  needed to answer all parts of the user's request, clearly and completely."
"If an answer cannot be found in context or any tool, say: \"I don't know.\""
"If greeted, respond warmly!"

            ),
        },
        *history
    ]

    # 1. Insert RAG context and user message
    context = retrieve_context(question, vectorstore)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    # 2. First pass: let LLM choose tools, if any
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024,
    )
    message = response.choices[0].message

    # 3. Handle tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_results = []
        for tool_call in message.tool_calls:
            fname = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = function_router(fname, args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Add tool call and tool result messages to chat history
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })
        messages.extend(tool_results)

        # 4. Second LLM pass for final answer (crucial! Don't return tool_call!)
        response_final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # or "none" to force LLM to only synthesize, not call more tools
            max_tokens=512,
        )
        return response_final.choices[0].message.content.strip()
    else:
        # No tool calls, just answer
        return message.content.strip()
