import streamlit as st
import langchain
from langchain.agents import create_csv_agent
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
from streamlit_chat import message
import requests
from bs4 import BeautifulSoup
import openai
from pathlib import Path 
from llama_index import download_loader, GPTVectorStoreIndex, LLMPredictor, QuestionAnswerPrompt, PromptHelper
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI

def main():
    st.sidebar.title("ChatMLS")  # Add title to the sidebar

    st.title("Talk with your MLS Data!")

    
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key to start", type="password")  # Add OpenAI API Key input field    

    openai.api_key = openai_api_key     

    # Create a sidebar menu
    #page = st.sidebar.selectbox("Select a page", ["Chat Support-MLS Import Wizard", "Chat Support-MLS Import Wizard", "Chat PDF", "Chat MLS"])    
    page = st.sidebar.selectbox("Select a page", ["Chat MLS"]) 

    with st.sidebar:
        st.markdown(
                "(Multiple Listings Service) MLS property data have information about homes that are sold or Listings for sale. It tells us where they are, how much they cost, how "
                "many bedrooms and bathrooms they have, and other important things about the property. It is used by real estate and " 
                "appraisal agents."
            )

    with st.sidebar:
        st.markdown(
                "This tool is a work in progress. "
                "ChatMLS an AI chatbot designed to help users explore and discuss about their MLS property data in a more intuitive way. " 
                "You can contribute to the project with your feedback and suggestions💡"
            )
        
    
    

    
    st.sidebar.text("Developed by Jeferson Tobias")  # Add title to the sidebar 
    st.sidebar.text("email : chatmls.ai@gmail.com")  # Add title to the sidebar   
    st.sidebar.text("version 1.0.1")  # Add title to the sidebar

    
    if page == "Chat MLS":
        pageMLS( openai_api_key )

def pageMLS( api_key ):
    try:
        if api_key is None or api_key == "":
            st.error("OPENAI_API_KEY is not set")
            return
        
        st.title("💬 Chat MLS ")
        st.write("Using : LangChain-Agent and OpenAI")

        csv_file = st.file_uploader("Upload a CSV file", type="csv")
        #csv_path = csv_file.name
        if csv_file is not None:
            csv_path = os.path.join(os.getcwd(), csv_file.name)  # Get the path of the uploaded CSV file
            with open(csv_path, "wb") as f:
                f.write(csv_file.read())  # Save the uploaded CSV file to the specified path
            agent = create_csv_agent(OpenAI(openai_api_key=api_key, temperature=0), csv_path, verbose=True)
            os.remove(csv_path)
            #agent = create_csv_agent(OpenAI(openai_api_key=api_key, temperature=0), csv_file, verbose=True)        
            #agent = create_csv_agent(OpenAI(openai_api_key=api_key, temperature=0), csv_file, verbose=True)       

            #user_question = st.text_input("Ask a question about your CSV: ")
            # show user input
            with st.form("csv_input_form", clear_on_submit=True):
                a, b = st.columns([4, 1])
                user_question = a.text_input(
                    label="Ask a question about your MLS:",
                    placeholder="What would you like to ask about your MLS ?",
                    label_visibility="collapsed",
                )
                b.form_submit_button("Send", use_container_width=True)
            
            #message('How can I assist you ?', is_user=False)  # Display AI's response 
            if user_question is not None and user_question != "":
                with st.spinner(text="Thinking..."):
                    prompt = user_question[:4000]  # Truncate the prompt to 4000 characters
                    completion_length = 700  # Set the desired completion length
                    completion = agent.run(prompt)[:completion_length]  # Truncate the completion to the desired length
                    with st.spinner(text="In progress..."):                    
                        message(prompt, is_user=True)
                        message(completion, is_user=False)  # Display AI's response
                    #st.write(completion)
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
