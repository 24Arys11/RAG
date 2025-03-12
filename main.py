from response_generator import OpenaiResponseGenerator
from data_retriever import DataRetriever
from rag import RAG


def start_chat(rag_system):
    user_query = input("User: ")
    while user_query.lower() != 'quit':
        response = rag_system.generate_response(user_query)
        print(f"Assistant: {response}")
        user_query = input("User: ")

def main():
    base_url, api_key = "http://127.0.0.1:1234/v1", "api_key"
    model = "qwen2.5-7b-instruct-1m"
    generator = OpenaiResponseGenerator(base_url, api_key, model, hystory_length=10)
    retriever = DataRetriever('data', chunk_size=512, top_k=5)

    rag_system = RAG(retriever, generator)

    start_chat(rag_system)

if __name__ == "__main__":
    main()