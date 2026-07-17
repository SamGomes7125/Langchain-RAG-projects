import os
from dotenv import load_dotenv
import yt_dlp
from groq import Groq
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ---- 1. Download audio from YouTube ----
def download_audio_from_youtube(urls, job_id):
    video_info = []
    for i, url in enumerate(urls):
        filename = f'{job_id}_{i}'
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '32',
            }],
            'postprocessor_args': ['-ac', '1', '-ar', '16000'],
            'quiet': True,
            'js_runtimes': {'node': {}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")
        video_info.append((f'{filename}.mp3', title, author))
    return video_info

urls = [
    "https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",
]
video_details = download_audio_from_youtube(urls, 1)
print(video_details)

# ---- 2. Transcribe each video with Groq's hosted Whisper ----
client = Groq()
results = []
for audio_file, title, author in video_details:
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            response_format="text",
        )
    results.append(transcription)
    print(f"Transcription for {title}:\n{transcription}\n")

# ---- 3. Chunk ALL videos' transcripts  ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

docs = []
for (audio_file, title, author), transcript in zip(video_details, results):
    chunks = text_splitter.split_text(transcript)
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata={"title": title, "author": author}))

print(f"Total chunks across all videos: {len(docs)}")

# ---- 4. Store in Deep Lake ----
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

my_activeloop_org_id = "your-activeloop-username"  
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

# ---- 5. Retrieval-QA chain ----
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {input}
Summarized answer in bullet points:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

document_chain = create_stuff_documents_chain(llm, PROMPT)
qa = create_retrieval_chain(retriever, document_chain)

result = qa.invoke({"input": "Summarize the mentions of google according to their AI program"})
print(result["answer"])
