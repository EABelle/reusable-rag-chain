# imports
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter


# load in the .env variables
load_dotenv()

class RagChain:

    class Reader:
        def __init__(self):
            self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

        def __read_file(self, path) -> str:
            with open(path) as f:
                return f.read()

        def create_chunks(self, path) -> list:
            text = self.__read_file(path)
            return self.text_splitter.split_text(text)

    class VectorStore:
        def __init__(self, embeddings):
            self.vector_store = Chroma(
                collection_name="my_collection",
                embedding_function=embeddings
            )
        def add_texts(self, texts):
            self.vector_store.add_texts(texts)

        def add_documents(self, documents):
            self.vector_store.add_documents(documents)

        def similarity_search(self, query, k=2):
            return self.vector_store.similarity_search(query, k=k)

    class Retriever:
        def __init__(self, vector_store):
            self.vector_store = vector_store

        def retrieve(self, query):
            return self.vector_store.similarity_search(query, k=2)

    def __init__(self):
        self.reader = self.Reader()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = self.VectorStore(self.embeddings)
        self.retriever = self.Retriever(self.vector_store)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.custom_rag_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        )
        self.rag_chain = (
            {"context": RunnableLambda(self.retriever.retrieve) | self.__format_docs, "query": RunnablePassthrough()}
            | self.custom_rag_prompt
            | self.llm
            | StrOutputParser()
        )
    def add_texts(self, texts):
        self.vector_store.add_texts(texts)
        
    def add_file(self, path):
        self.vector_store.add_texts(
            self.reader.create_chunks(path)
        )

    def query(self, query_text):
        results = self.vector_store.similarity_search(query_text, k=2)
        for res in results:
            print(f"* {res.page_content} [{res.metadata}]\n\n")
        return results

    def invoke(self, query_text):
        return self.rag_chain.invoke(query_text)

    def __format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
