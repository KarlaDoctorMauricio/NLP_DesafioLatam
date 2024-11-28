from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from dotenv import load_dotenv

import glob
import logging
import os


load_dotenv()
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def load_data(directory_path, files, txt_files):
  """
  This script reads text files, extracts metadata (title and URL) from a CSV, 
  and stores the content and metadata in Document objects for further processing. 
  Errors during processing are logged.
  """
  # Initialize a list to store documents
  data = []
  # Process each .txt file
  for txt_file in txt_files:
      try:
          # Read the content of the text file
          with open(txt_file, 'r', encoding='utf-8') as file:
              content = file.read()  # Read the content from the file
              
              # Extract the corresponding file name and URL from the metadata CSV
              file_name = files.iloc[int(txt_file[len(directory_path) + 1:-4])]['title']
              url = files.iloc[int(txt_file[len(directory_path) + 1:-4])]['url']
              
              # Append a Document object with content and metadata (source and URL)
              data.append(Document(page_content=content, metadata={"source": file_name, "url": url}))
              
              # Optionally print the file name for logging purposes
              print(file_name)
      
      except Exception as e:
          # Log an error message in case of issues while processing a file
          logging.error(f"Error processing the file {txt_file}. Exception: {e}")
  return data

def chunking(data):
  """
  Splits the input documents into smaller chunks of text with a specified 
  maximum chunk size and overlap. Returns the list of text chunks.
  """
  # Define the text splitter with a chunk size of 300 characters and an overlap of 100 characters
  text_splitter    = RecursiveCharacterTextSplitter(
      chunk_size   = 300,    # Max number of characters per chunk
      chunk_overlap= 100     # Number of characters overlapping between consecutive chunks
  )

  # Split the documents into smaller chunks
  docs = text_splitter.split_documents(data)

  # Print the total number of documents (chunks) created
  #print("Total number of documents: ", len(docs))
  return docs

def process_and_store_embeddings(data, embeddings):
  """
  Processes the input documents by splitting them into smaller chunks, 
  computes their embeddings using the provided embedding model, 
  and stores the resulting embeddings in a Chroma vector database for efficient retrieval.
  """
  docs = chunking(data)
  ## Embeddings (text -> numerics)
  # Initialize the embedding model from Google Generative AI
  #Add embeddings to a vector database (Chroma)
  vectorstore = Chroma.from_documents(
      documents = docs,
      embedding = embeddings,
      persist_directory="chroma_db"  
      )
  
def retrieve_documents(query, vectorstore):
  """
    Retrieves the most relevant documents for the given query using similarity search.
  """
    
  retriever = vectorstore.as_retriever(
      search_type="similarity",  # Type of search: similarity search
      search_kwargs={"k": 3},  # Return the top 3 similar documents
      embedding_function=embeddings  # The function used to compute embeddings for the query
  )

  # Use the retriever to get the most relevant documents for the query
  retrieved_docs = retriever.invoke(query)  # Invoke the retriever with the query

  # Loop through the retrieved documents and print their content and metadata
  for doc in retrieved_docs:
      content = doc.page_content  # Get the content of the document
      source = doc.metadata.get("source", "Unknown source")  # Get the 'source' metadata of the document, defaulting to "Unknown source" if not found
      
      # Print the source and content of the document
      print(f"Source: {source}")
      print(f"Content: {content}")
      print("-" * 50)  # Print a separator line for readability
  return retriever

def generate_answer_with_rag(retriever, query):
  """
  Generates an answer to the provided query using Retrieval-Augmented Generation (RAG) technique.
  This function combines the power of a pre-trained language model (Google Generative AI) and a retrieval system
  to generate responses based on retrieved context from documents.    
  """
  # Initialize the language model (llm) with Google Generative AI, specifying model and parameters
  llm =  ChatGoogleGenerativeAI(
      model = "gemini-1.5-pro",  # Model name
      temperature = 0.7,         # Control the randomness of the output (higher value means more random)
      max_tokens = 100           # Limit the length of the generated response
  )
  # System prompt to guide the assistant in answering questions
  system_prompt = (
      """Eres un asistente para tareas de preguntas y respuestas. 
      Utiliza los siguientes fragmentos de contexto recuperados para responder a la pregunta. 
      Si no sabes la respuesta, indica que no la sabes. Usa un máximo de tres oraciones y mantén la respuesta concisa.
      \n\n"
      "{context}"""
  )

  # Create the prompt template by combining system prompt and user input
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),  # System's instruction
          ("human", "{input}"),       # User's input
      ]
  )

  # Create the chain for processing documents and generating answers (using retrieval-augmented generation)
  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  

  # Create the RAG chain that combines retrieval and generation for answering the question
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  response = rag_chain.invoke({"input": query})
  return response


def RAG(query, files, directory_path):
  """
  This function implements a Retrieval-Augmented Generation (RAG) process to
  answer queries based on course content stored in text files and metadata.

  """
  #### Load data ####
  # Define the directory containing the course content files
  txt_files = glob.glob(f"{directory_path}/*.txt")  # Get all .txt files in the specified directory

  print("🔵🔵 1. Load data")
  #Load data in a Document object
  data = load_data(directory_path, files, txt_files)

  #### Load vectorstore ####
  directory_db = "chroma_db"
  if not os.path.exists(directory_db):
    os.makedirs(directory_db)

  if not os.listdir(directory_db):
    process_and_store_embeddings(data, embeddings)

  vectorstore = Chroma(persist_directory=directory_db, embedding_function = embeddings)

  print("🔵🔵 2. Retriever documents")
  retriever = retrieve_documents(query, vectorstore)

  #### Generation ####
  print("🔵🔵 3. Generative answer *complete RAG")
  response = generate_answer_with_rag(retriever, query)

  recommended_resources = []
  for i in range(len(response)):
    element = [response["context"][i].metadata['source'], response["context"][i].metadata['url']]
    if element not in recommended_resources:
      recommended_resources.append(element)

  return [response["answer"], recommended_resources]