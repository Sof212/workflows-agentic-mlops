
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class struct_python(BaseModel):
    python_code: str=Field(
        description="only the python code"
    )
    description: str= Field(description="the reste of the message")
    requirements: List[str]= Field(description="mettre toutes requiremnts a installer")

class CodeGeneration:
    """
    A class for generating and reviewing Python and SQL code
    using an OpenAI language model.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

        self.chat_model = ChatOpenAI(
            model=self.model,
            temperature=0.0
        ).with_structured_output(struct_python)

    @staticmethod
    def synthesise_responses(response: str) -> str:
        """Clean and return the generated response."""
        return response.strip()

    @staticmethod
    def generate_prompt(query: str) -> str:
        """Generate a structured prompt for code generation."""

        prompt = f"""
You are an expert software engineer specialized in
Python and SQL code generation and code review.

Your mission is to generate optimized, efficient,
well-structured, and documented code.

Follow these requirements:

1. Ensure the code is syntactically correct.
2. Optimize performance and efficiency.
3. Include clear documentation and comments.
4. Follow clean code principles.
5. Handle potential errors and edge cases.
6. Provide explanations when necessary.

Example:

User query:
Generate a Python function that returns the sum
of all even numbers in a list.

Expected output:

```python
def sum_of_even_numbers(numbers: list[int]) -> int:
    '''Return the sum of all even integers.'''
    return sum(number for number in numbers if number % 2 == 0)
```

Now process this user request:

{query}

Return clean, executable code.
"""
        return prompt

    def generate_code(self, query: str) -> str:
        """Generate code based on the user's query."""

        prompt = self.generate_prompt(query)

        response = self.chat_model.invoke(prompt)

        return response


