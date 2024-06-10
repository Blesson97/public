import os
import uuid
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import clean_and_tokenize

def clone_repository(github_url, local_path):
    """
    Clones a GitHub repository to the specified local path.
    
    Args:
        github_url (str): The URL of the GitHub repository.
        local_path (str): The local path to clone the repository.

    Returns:
        bool: True if cloning is successful, False otherwise.
    """
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False

def load_files(repo_path):
    """
    Loads and indexes files from a repository.

    Args:
        repo_path (str): The path to the repository.

    Returns:
        tuple: A tuple containing the index, split documents, file type counts, and sources of the split documents.
    """
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']
    file_type_counts = {}
    documents_dict = {}

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext == 'ipynb':
                loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = get_loaded_documents(loader)
            update_file_type_counts(loaded_documents, file_type_counts, ext)
            update_documents_dict(loaded_documents, documents_dict, repo_path)
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    split_documents = get_split_documents(documents_dict)
    index = get_index(split_documents)

    sources = get_document_sources(split_documents)
    return index, split_documents, file_type_counts, sources

def get_loaded_documents(loader):
    """
    Loads documents from a loader.

    Args:
        loader: The document loader.

    Returns:
        list: List of loaded documents.
    """
    if callable(loader.load):
        return loader.load()
    else:
        return []

def update_file_type_counts(loaded_documents, file_type_counts, ext):
    """
    Updates the file type counts dictionary.

    Args:
        loaded_documents (list): List of loaded documents.
        file_type_counts (dict): Dictionary to store file type counts.
        ext (str): File extension.
    """
    if loaded_documents:
        file_type_counts[ext] = len(loaded_documents)

def update_documents_dict(loaded_documents, documents_dict, repo_path):
    """
    Updates the documents dictionary.

    Args:
        loaded_documents (list): List of loaded documents.
        documents_dict (dict): Dictionary to store the documents.
        repo_path (str): The path to the repository.
    """
    for doc in loaded_documents:
        file_path = doc.metadata['source']
        relative_path = os.path.relpath(file_path, repo_path)
        file_id = str(uuid.uuid4())
        doc.metadata['source'] = relative_path
        doc.metadata['file_id'] = file_id

        documents_dict[file_id] = doc

def get_split_documents(documents_dict):
    """
    Splits the original documents into smaller documents.

    Args:
        documents_dict (dict): Dictionary of original documents.

    Returns:
        list: List of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    split_documents = []

    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']

        split_documents.extend(split_docs)

    return split_documents

def get_index(split_documents):
    """
    Creates an index from split documents.

    Args:
        split_documents (list): List of split documents.

    Returns:
        BM25Okapi: The index.
    """
    if split_documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
        index = BM25Okapi(tokenized_documents)
        return index
    else:
        return None

def get_document_sources(split_documents):
    """
    Gets the sources of split documents.

    Args:
        split_documents (list): List of split documents.

    Returns:
        list: List of document sources.
    """
    sources = [doc.metadata['source'] for doc in split_documents]
    return sources

def search_documents(query, index, documents, n_results=5):
    """
    Searches for documents based on a query and returns the top results.

    Args:
        query (str): The query string.
        index: The index used for searching.
        documents: The list of documents.
        n_results (int): The number of top results to return.

    Returns:
        list: The top search results.
    """
    query_tokens = clean_and_tokenize(query)
    bm25_scores = index.get_scores(query_tokens)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    query_tfidf = tfidf_vectorizer.transform([query])

    cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5

    unique_top_document_indices = get_unique_top_document_indices(combined_scores, n_results)

    return get_top_documents(documents, unique_top_document_indices)

def get_unique_top_document_indices(combined_scores, n_results):
    """
    Gets the unique top document indices.

    Args:
        combined_scores (numpy.ndarray): Array of combined scores.
        n_results (int): The number of top results to return.

    Returns:
        list: List of unique top document indices.
    """
    unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]
    return unique_top_document_indices

def get_top_documents(documents, unique_top_document_indices):
    """
    Gets the top documents based on the unique top document indices.

    Args:
        documents (list): List of documents.
        unique_top_document_indices (list): List of unique top document indices.

    Returns:
        list: List of top documents.
    """
    return [documents[i] for i in unique_top_document_indices]