import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle

class DataRetriever:
    def __init__(self, directory_path, chunk_size=512, top_k=5):
        self.directory_path = directory_path
        if not os.path.isdir(self.directory_path):
            raise ValueError(f"Directory does not exist: {self.directory_path}")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.index_path = 'faiss_index'
        self.embeddings_path = 'embeddings.pkl'
        self.document_chunks = self._chunk_documents()
        self.index, self.embeddings = self._load_or_build_index()

    def _chunk_documents(self):
        documents = [f for f in os.listdir(self.directory_path) if f.endswith('.txt')]
        if not documents:
            raise ValueError("No documents found in the directory.")

        document_chunks = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".txt"):
                cleaned_text = self._clean_text_file(os.path.join(self.directory_path, filename))
                chunked_text = [cleaned_text[i:i + self.chunk_size] for i in range(0, len(cleaned_text), self.chunk_size)]
                document_chunks.extend(chunked_text)
        return document_chunks

    def _clean_text_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        cleaned_text = text.replace('\n', ' ').replace('\r', '').strip()
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    def retrieve(self, query):
        query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
        distances, indices = self.index.search(query_embedding, self.top_k)
        relevant_texts = [f"... {self.document_chunks[idx]} ..." for idx in indices[0]]
        return '\n'.join(relevant_texts)

    def _load_or_build_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            index = faiss.read_index(self.index_path)
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            return index, embeddings
        else:
            return self._build_and_save_index()

    def _build_and_save_index(self):
        embeddings = self.model.encode(
            self.document_chunks, convert_to_tensor=True
        ).cpu().numpy()

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        return index, embeddings
