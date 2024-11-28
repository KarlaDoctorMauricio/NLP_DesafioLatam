# Retrieval-Augmented Generation (RAG) Workflow

This project demonstrates how to use a **Retrieval-Augmented Generation (RAG)** approach to answer questions based on a collection of course content stored in text files. The system combines **document retrieval** using a vector store and **text generation** with the **Google Generative AI** model to provide context-aware answers to user queries.


*Special thanks to the original authors of the https://github.com/AarohiSingla/Generative_AI/blob/main/L-8/gemini_rag_demo/basics_RAG_pdf.ipynb  notebook for providing the foundation of this project

## Requirements
Before running the script, make sure you have the following installed:

- Python 3.x
- Required Python libraries (are installed within the code)
- For the scraping section*

*Directly check the folder to install the necessary libraries. The idea here is that you should be able to run the RAG with text files, regardless of whether they come from scraping

## Setup

### 1. Environment Variables
Create a `.env` file in the root directory of the project to store sensitive information, such as API keys, if required.

Example `.env` file:

On this site, you can generate that API with your account: https://aistudio.google.com/app/u/1/apikey?pli=1


## Functions

### 1. `load_data(directory_path, files, txt_files)`
This function loads text files from a specified directory, reads their content, and retrieves associated metadata (e.g., title, URL) from a CSV file. It stores the content and metadata in `Document` objects for further processing.

#### Parameters:
- `directory_path`: Path to the directory containing the `.txt` files.
- `files`: Pandas DataFrame containing metadata (titles, URLs).
- `txt_files`: List of paths to `.txt` files.

#### Returns:
- A list of `Document` objects, each containing content and metadata.

---

### 2. `chunking(data)`
This function splits the documents into smaller chunks using a text splitter. The chunks are created with a maximum size of 300 characters and an overlap of 100 characters between consecutive chunks.

#### Parameters:
- `data`: List of `Document` objects to be chunked.

#### Returns:
- A list of smaller text chunks.

---

### 3. `process_and_store_embeddings(data, embeddings)`
This function processes the input documents by chunking them and then generates embeddings using the specified model (`GoogleGenerativeAIEmbeddings`). The resulting embeddings are stored in a Chroma vector database for efficient retrieval.

#### Parameters:
- `data`: List of `Document` objects.
- `embeddings`: The embedding model to use for generating embeddings.

#### Returns:
- None (the embeddings are stored in the `chroma_db` directory).

---

### 4. `retrieve_documents(query, vectorstore)`
This function performs a **similarity search** to retrieve the most relevant documents for a given query. It returns the top 3 documents based on their similarity to the query.

#### Parameters:
- `query`: The user input/query.
- `vectorstore`: The Chroma vector database containing document embeddings.

#### Returns:
- A list of the top 3 most relevant documents.

---

### 5. `generate_answer_with_rag(retriever, query)`
This function generates an answer to the user's query using the **Retrieval-Augmented Generation (RAG)** approach. It retrieves the relevant documents and then generates a concise response based on the retrieved context using the **Google Generative AI model**.

#### Parameters:
- `retriever`: The document retriever (Chroma vectorstore).
- `query`: The user's input query.

#### Returns:
- A generated response in the form of a JSON-like structure, including the answer and recommended resources.

---

### 6. `RAG(query, files, directory_path)`
This is the main function that orchestrates the entire RAG pipeline:
1. Loads data from text files and metadata.
2. Processes and stores document embeddings in a Chroma vector store.
3. Retrieves relevant documents for the given query.
4. Generates an answer based on the retrieved documents using the RAG approach.

#### Parameters:
- `query`: The user's input query.
- `files`: A Pandas DataFrame containing metadata for the documents (e.g., title, URL).
- `directory_path`: The path to the directory containing the `.txt` files.

#### Returns:
- A tuple containing:
  1. The generated answer to the query.
  2. A list of recommended resources (source and URL of relevant documents).

---

## File Structure

>     project-directory/  
>     │  
>     ├── chroma_db/                   # Folder with the vector database     
>           ├──chroma.sqlite3 
>     ├── Scrapping                    # Directory with scraping results      
>     ├── RAG_gemini.py                # RAG implementation based on Gemini 
>     ├── main.ipynb                   # Example of RAG running         
>     └── requierements.txt            # Library with the neccesary libraries



Notes
-----
- This is an example adapted for the DesafioLatam 2024 Hackathon, so the necessary adaptations will need to be made for it to work.

License
-------
![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.