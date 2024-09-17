import os
import streamlit as st
from io import StringIO
import re
import sys

import pickle
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ! chatbot 

import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

# Fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False

# * Embedder

class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, original_filename):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
            
        def get_file_extension(uploaded_file):
            file_extension =  os.path.splitext(uploaded_file)[1].lower()
            
            return file_extension
        
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 2000,
                chunk_overlap  = 100,
                length_function = len,
            )
        
        file_extension = get_file_extension(original_filename)

        if file_extension == ".csv":
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8",csv_args={
                'delimiter': ',',})
            data = loader.load()

        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)  
            data = loader.load_and_split(text_splitter)
        
        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split(text_splitter)
            
        embeddings = OpenAIEmbeddings()

        vectors = FAISS.from_documents(data, embeddings)
        os.remove(tmp_file_path)

        # ! old: Save the vectors to a pickle file
        #with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            #pickle.dump(vectors, f)
        
        # * new: 
        with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
        # Use the highest protocol available
            pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)


    def getDocEmbeds(self, file, original_filename):
        """
        Retrieves document embeddings
        """
        if not os.path.isfile(f"{self.PATH}/{original_filename}.pkl"):
            self.storeDocEmbeds(file, original_filename)

        # Load the vectors from the pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "rb") as f:
            vectors = pickle.load(f)
        
        return vectors

# * from modules.utils import Utilities

# ! ---- 

class Utilities:

    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or 
        from the user's input and returns it
        """
        if not hasattr(st.session_state, "api_key"):
            st.session_state.api_key = None
        #you can define your API key in .env directly
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            if st.session_state.api_key is not None:
                user_api_key = st.session_state.api_key
                st.sidebar.success("API key loaded from previous input", icon="🚀")
            else:
                user_api_key = st.sidebar.text_input(
                    label="#### Your OpenAI API key 👇", placeholder="sk-...", type="password"
                )
                if user_api_key:
                    st.session_state.api_key = user_api_key

        return user_api_key

    
    @staticmethod
    def handle_upload(file_types):
        """
        Handles and display uploaded_file
        :param file_types: List of accepted file types, e.g., ["csv", "pdf", "txt"]
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=file_types, label_visibility="collapsed")
        if uploaded_file is not None:

            def get_file_extension(uploaded_file):
                return os.path.splitext(uploaded_file)[1].lower()
            
            # file_extension = get_file_extension(uploaded_file.name)

        else:
            st.session_state["reset_chat"] = True

        #print(uploaded_file)
        return uploaded_file

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()

        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)

            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature,vectors)
        st.session_state["ready"] = True

        return chatbot

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    qa_template = """
        You are SpEdGPT, a specialized AI assistant designed to help users understand and simplify complex formulas. 
        When provided with a formula, use the context given (from the PDF file) to provide a clear and accurate explanation. 
        Your goal is to help the user (developer, quality analyst, product manager and support engineers) understand the purpose and logic of the formula.
        
        Have 3 sections for the provided answer labeled as 'Formula Simplification', 'Business Purpose', 'Summary'. 
        'Formula Simplification' - will have the technical explanation, break down function parameters and their data types for better understanding (use paranthesis to segregate response where needed).
        Explain business purpose in-terms of decision making.
        Summary should contain a single paragraph (no further breakdowns).
        
        If you do not know the answer, honestly state that you do not know. Do not attempt to fabricate an answer.
        
        If a question falls outside the scope of the provided context, kindly inform the user that you are focused on addressing questions specifically related to the given context.
        
        Provide as much detail and clarity as possible in your responses to ensure thorough understanding.

        Responses should contain all the requirements declared above with minimal labeling and bold.

        If user asks to generating something new on top of the formula, please convey your apology as you're restricted only to simplify the understanding of the formula and not generating something new to avoid any kind of possibility to breakdown the existing business functionality served by the formula.

        context: {context}
        =========
        question: {question}
        ======
    """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    def load_pdf_content(self):
        """
        Load the default PDF file and extract its text content
        """
        content = ""
        try:
            with uploaded_file as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text()
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
        return content

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        # Load PDF content to provide context
        context = self.load_pdf_content()

        # ! old: Updated the verbose argument to False to avoid the attribute error
        #chain = ConversationalRetrievalChain.from_llm(
            #llm=llm,
            #retriever=retriever,
            #verbose=False,  # Changed verbose to False
            #return_source_documents=True,
            #max_tokens_limit=4097,
            #combine_docs_chain_kwargs={'prompt': self.QA_PROMPT}
        #)

        # * new: 
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=False,  # Changed verbose to False to avoid attribute error
            return_source_documents=True,
            max_tokens_limit=4097,
            combine_docs_chain_kwargs={'prompt': self.QA_PROMPT}
        )


        chain_input = {"context": context, "question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        # count_tokens_chain(chain, chain_input)
        return result["answer"]


# ! chatbot 

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation: {cb.total_tokens} tokens')
    return result


# * from modules.history import ChatHistory

from streamlit_chat import message

class ChatHistory:
    def __init__(self):
        self.history = st.session_state.get("history", [])
        st.session_state["history"] = self.history

    def default_prompt(self, topic):
        return f"Please enter any formula for better understanding! 🤗"

    def initialize_assistant_history(self, uploaded_file):
        st.session_state["assistant"] = [self.default_prompt(uploaded_file.name)]
    
    def initialize_user_history(self):
        st.session_state["user"] = []  # Initialize with an empty list

    def initialize(self, uploaded_file):
        if "assistant" not in st.session_state:
            self.initialize_assistant_history(uploaded_file)
        if "user" not in st.session_state:
            self.initialize_user_history()

    def reset(self, uploaded_file):
        st.session_state["history"] = []
        self.initialize_user_history()
        self.initialize_assistant_history(uploaded_file)
        st.session_state["reset_chat"] = False

    def append(self, mode, message):
        st.session_state[mode].append(message)

    def generate_messages(self, container):
        if st.session_state.get("assistant"):
            with container:
                for i, assistant_message in enumerate(st.session_state["assistant"]):
                    message(assistant_message, key=f"assistant_{i}", avatar_style="thumbs", is_user=False)
                    if "user" in st.session_state and i < len(st.session_state["user"]):
                        user_message = st.session_state["user"][i]
                        # Set `is_user` to False for left alignment
                        message(user_message, key=f"user_{i}", avatar_style="big-smile", is_user=False)

    def load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                self.history = f.read().splitlines()

    def save(self):
        with open(self.history_file, "w") as f:
            f.write("\n".join(self.history))


# * from modules.layout import Layout

class Layout:
    
    def show_header(self, types_files):
        """
        Displays the header of the app
        """
        st.markdown(
            f"""
            <h2 style='text-align: center;'> Ask SpEdGPT about your Formulas!</h2>
            """,
            unsafe_allow_html=True,
        )

    def show_api_key_missing(self):
        """
        Displays a message if the user has not entered an API key
        """
        st.markdown(
            """
            <div style='text-align: center;'>
                <h4>Enter your <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API key</a> to start chatting</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def prompt_form(self):
        """
        Displays the prompt form
        """
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="Ask me anything related to the formula...",
                key="input",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(label="Send")
            
            is_ready = submit_button and user_input
        return is_ready, user_input
    




# * from modules.sidebar import Sidebar

class Sidebar:

    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("🧠 About SpEdGPT ")
        sections = [
            "#### SpEdGPT is an AI chatbot with a conversational memory, designed to allow users to discuss their data in a more intuitive way. 📄",
            "#### It uses large language models to provide users with natural language interactions about user data content. 🌐",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature

        # * Display a message that updates based on the temperature value
        if temperature == 0:
            st.write("Temperature is set to 0")
        else:
            st.write(f"Temperature is set to {temperature}")
        
    def show_options(self):
        with st.sidebar.expander("🛠️ SpEdGPT's Tools", expanded=False):

            self.reset_chat_button()
            self.model_selector()
            self.temperature_slider()
            st.session_state.setdefault("model", self.MODEL_OPTIONS[1])
            st.session_state.setdefault("temperature", self.TEMPERATURE_DEFAULT_VALUE) 

    

# Config
st.set_page_config(layout="wide", page_icon="💬", page_title="SpEdGPT | Formula Simplifier 🤖")

# Contact
with st.sidebar.expander("📬 Contact"):
    st.write("**PowerSchool - Special Programs**")

# Title and Special Programs text
st.markdown(
    """
    <div style="flex: 3; text-align: center;">
        <h2>🚀 SpEdGPT, Special Programs Formula Assistant 🤖</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Description
#st.markdown(
    #""" 
    #<h5 style='text-align:center;'>I'm SpEdGPT, ready to help you better understand your formulas.</h5>
    #""",
    #unsafe_allow_html=True,
#)
#st.markdown("---")

# SpEdGPT's Intelligence
#st.subheader("🚀 SpEdGPT's Intelligence")
#st.write("""
#- **SpEdGPT-Chat**: General Chat on data (PDF, TXT, CSV) with a vectorstore | works with ConversationalRetrievalChain
#""")
#st.markdown("---")

#To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

# history_module = reload_module('history')
# layout_module = reload_module('layout')
# utils_module = reload_module('utils')
# sidebar_module = reload_module('sidebar')

# ChatHistory = history_module.ChatHistory
# Layout = layout_module.Layout
# Utilities = utils_module.Utilities
# Sidebar = sidebar_module.Sidebar

# st.set_page_config(layout="wide", page_icon="💬", page_title="SpEdGPT | Chat-Bot 🤖")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

layout.show_header("PDF, TXT, CSV")

user_api_key = utils.load_api_key()

if not user_api_key:
    layout.show_api_key_missing()
else:
    os.environ["OPENAI_API_KEY"] = user_api_key

    uploaded_file = utils.handle_upload(["pdf", "txt", "csv"])

    if uploaded_file:

        # Configure the sidebar
        sidebar.show_options()
        sidebar.about()

        # Initialize chat history
        history = ChatHistory()
        try:
            chatbot = utils.setup_chatbot(
                uploaded_file, st.session_state["model"], st.session_state["temperature"]
            )
            st.session_state["chatbot"] = chatbot

            if st.session_state["ready"]:
                # Create containers for chat responses and user prompts
                response_container, prompt_container = st.container(), st.container()

                with prompt_container:
                    # Display the prompt form
                    is_ready, user_input = layout.prompt_form()

                    # Initialize the chat history
                    history.initialize(uploaded_file)

                    # Reset the chat history if button clicked
                    if st.session_state["reset_chat"]:
                        history.reset(uploaded_file)

                    if is_ready:
                        # Update the chat history and display the chat messages
                        history.append("user", user_input)

                        old_stdout = sys.stdout
                        sys.stdout = captured_output = StringIO()

                        output = st.session_state["chatbot"].conversational_chat(user_input)

                        sys.stdout = old_stdout

                        history.append("assistant", output)

                        
                            
                history.generate_messages(response_container)
        except Exception as e:
            st.error(f"Error: {str(e)}")


