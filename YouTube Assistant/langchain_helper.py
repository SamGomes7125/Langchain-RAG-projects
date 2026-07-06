from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate  
from langchain_community.vectorstores import DocArrayInMemorySearch  # <-- 1. Changed to DocArray
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the Google Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

def create_vectordb_from_youtube(youtube_url: str) -> DocArrayInMemorySearch:
    # Load the YouTube video transcript
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    # <-- 2. Changed to DocArray initialization here
    db = DocArrayInMemorySearch.from_documents(docs, embeddings) 
    return db
    
def get_response_from_query(db, query, k=3): 
    docs = db.similarity_search(query, k=k)
    docs_page_contents = " ".join([doc.page_content for doc in docs])
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful YouTube assistant. 
        Based on the transcript of the video, answer the following question: {question} 
        by searching the context of the video transcript: {docs}
        
        Only use factual information. If the answer is not found in the transcript, respond with 'I don't know'. 
        Your answer should be detailed and informative."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"question": query, "docs": docs_page_contents})
    
    return response.content.replace("\n", " ").strip()