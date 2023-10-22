from utils import *
import json

# ingest PDF files
from langchain.document_loaders import PyPDFLoader
# Load GOOG's 10K annual report (92 pages).
url = "https://abc.xyz/investor/static/pdf/20230203_alphabet_10K.pdf"
loader = PyPDFLoader(url)
documents = loader.load()

# from google.colab import auth as google_auth
# google_auth.authenticate_user()

PROJECT_ID = ""
LOCATION = "us-central1"


vector_save_directory = 'D:\\Documents\\NotesHelper' #CHANGE THIS

# Store docs in local vectorstore as index
# it may take a while since API is rate limited
from langchain.vectorstores import Chroma

vector_read_from_db = Chroma(persist_directory=vector_save_directory,
                             embedding_function=embeddings)

# Expose index to the retriever
retriever = vector_read_from_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# Create chain to answer questions
from langchain.chains import RetrievalQA

# Uses LLM to synthesize results from the search index.
# We use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
              llm=llm,
              chain_type="stuff",
              retriever=retriever,
              return_source_documents=True
)

query = "What was Alphabet's net income in 2022?"
result = qa({"query": query})
print(result["result"])
print("Sources: ")
sources = result["source_documents"]
for doc in sources:
    print("Page: ", doc.metadata['page'])
    print("Page Content: ", doc.page_content)
