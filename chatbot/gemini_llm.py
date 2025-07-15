from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation
from typing import Literal

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(LLM):
    model_name: Literal["models/gemini-1.5-flash", "models/gemini-pro"] = "models/gemini-1.5-flash"

    def _call(self, prompt: str, stop=None) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"
