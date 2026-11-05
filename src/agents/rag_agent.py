
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

load_dotenv()


# 1. Define structured response
class QueryRequest(BaseModel):
    responses: str = Field(
        description="Answer to the user's question based on the documents"
    )

    Boolean: bool = Field(
        description="True if the answer is supported by the documents, False otherwise"
    )


# 2. Research Agent
class ResearchAgent:

    def __init__(self, query: str, name_model_llm: str="gpt-4o-mini"):
        self.query = query
        self.name_model_llm = name_model_llm

        self.chat_model = ChatOpenAI(
            model=self.name_model_llm,
            temperature=0.1,
            max_tokens=2000
        )


      # Enforce structured output
        self.structured_llm = self.chat_model.with_structured_output(
            QueryRequest
        )

    # 3. Synthesize responses
    def synthesize_responses(self, responses: str):

        return responses.strip()

    # 4. Generate prompt
    def generate_prompt(self, context: str):

        prompt = f"""
You are a helpful assistant specialized in RAG.

Your mission is to answer the question using exclusively
the information contained in the provided documents.

Rules:
- Be concise and precise.
- Do not invent information.
- If the answer cannot be found in the documents,
  respond with "I don't know".
- Boolean must be True only if the answer is supported
  by the provided documents.
- Boolean must be False otherwise.

Question:
{self.query}

Context:
{context}
"""
        return prompt

    # 5. Generate answer
    def generate_answer(
        self,
        query: str,
        documents: List[Document]
    ):

        if not documents:
            return QueryRequest(
                responses="I don't know",
                Boolean=False
            )

        context = "\n\n".join([
            f"Section: {doc.metadata.get('Header 2', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in documents
        ])

        prompt = self.generate_prompt(query, context)

        response = self.structured_llm.invoke(prompt)

        return response
