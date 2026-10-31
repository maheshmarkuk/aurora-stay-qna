from langchain_openai import ChatOpenAI
import os

def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

def generate(query, retrieved_chunks, model) -> str:
    prompt = f"""You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    - Use only information found in the retrieved context chunks.
    - If relevant information is missing, say "The answer is not available in the hotel policy documents provided."
    - Do not make up facts, policies, or procedures.
    - Clearly cite any special conditions or exceptions from the chunks.
    - Be concise, direct, and clear.
    """
    return model.invoke(prompt)

def rag(query: str, k: int = 2, model_name: str = "gpt-4o-mini", temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512, vectorstore=None):
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
