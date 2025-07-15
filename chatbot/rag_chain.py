from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .gemini_llm import GeminiLLM

def create_qa_chain(vectorstore):
    """
    Create a RetrievalQA chain using Gemini and a prompt template.
    :param vectorstore: Vector store to retrieve relevant chunks.
    :return: Configured RetrievalQA chain.
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = GeminiLLM()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

