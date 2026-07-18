import os
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from elevenlabs.client import ElevenLabs
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from streamlit_chat import message
from groq import Groq

load_dotenv()

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

my_activeloop_org_id = "swarnabhaghosh2005"
my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
dataset_path = f'hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}'

groq_api_key = os.environ.get('GROQ_API_KEY')
eleven_api_key = os.environ.get('ELEVENLABS_API_KEY')
elevenlabs_client = ElevenLabs(api_key=eleven_api_key)

def load_embeddings_and_database(active_loop_data_set_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = DeepLake(
        dataset_path=active_loop_data_set_path,
        read_only=True,
        embedding_function=embeddings
    )
    return db

def transcribe_audio(audio_file_path, groq_client):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        return transcription
    except Exception as e:
        print(f"Error calling Groq Whisper API: {str(e)}")
        return None

def record_and_transcribe_audio(groq_client):
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)
        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)
        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH, groq_client)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)
    return transcription

def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")

def get_user_input(transcription):
    if transcription and "input" not in st.session_state:
        st.session_state["input"] = transcription
    return st.text_input("", key="input")

def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['k'] = 4
    model = ChatGroq(model_name='llama-3.3-70b-versatile', temperature=0)
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa.invoke({'query': user_input})

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))
        text = history["generated"][i]
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_multilingual_v2",
        )
        audio_bytes = b"".join(audio)
        st.audio(audio_bytes, format='audio/mp3')

def main():
    st.write("# JarvisBase 🧙")
    groq_client = Groq()
    db = load_embeddings_and_database(dataset_path)
    transcription = record_and_transcribe_audio(groq_client)
    user_input = get_user_input(transcription)

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()