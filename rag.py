from data_retriever import DataRetriever
from response_generator import IResponseGenerator

class RAG:
    def __init__(self, retriever: DataRetriever, generator: IResponseGenerator):
        self.retriever = retriever
        self.generator = generator

    def generate_response(self, user_query):
        retrieved_docs = self.retriever.retrieve(user_query)

        intro = "Given some relevant information from the documents:"
        pre_prompt = "Answer to the following user request:"
        full_prompt = f"{intro}\n```\n{retrieved_docs}\n```\n{pre_prompt}\n{user_query}"

        response = self.generator.query(full_prompt)

        return response
