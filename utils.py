# utils.py

import re
import os
from nltk.tokenize import word_tokenize

def clean_and_tokenize(text):
    """
    Cleans and tokenizes the text.
    - Removes HTML tags, square brackets, parentheses, URLs, non-alphanumeric characters,
    numbers, and converts the text to lowercase.
    """
    cleaned_text = _remove_html_tags(text)
    cleaned_text = _remove_square_brackets(cleaned_text)
    cleaned_text = _remove_parentheses(cleaned_text)
    cleaned_text = _remove_urls(cleaned_text)
    cleaned_text = _remove_non_alphanumeric(cleaned_text)
    cleaned_text = _remove_numbers(cleaned_text)
    cleaned_text = cleaned_text.lower()
    return word_tokenize(cleaned_text)


def format_documents(documents):
    """
    Formats the list of documents.
    - Returns a numbered list with the basename of the source file and the page content of each document.
    """
    formatted_docs = []
    for doc in documents:
        formatted_doc = format_document(doc)
        formatted_docs.append(formatted_doc)
    return "\n".join(formatted_docs)


def format_document(document):
    """
    Formats a single document.
    - Returns the basename of the source file and the page content of the document.
    """
    basename = get_basename(document.metadata['source'])
    page_content = document.page_content
    return f"{basename}: {page_content}"


def format_user_question(question):
    """
    Formats the user's question.
    - Removes excessive whitespace.
    """
    return remove_excessive_whitespace(question)


def _remove_html_tags(text):
    """
    Removes HTML tags from the text.
    """
    return re.sub(r'<[^>]*>', '', text)


def _remove_square_brackets(text):
    """
    Removes square brackets from the text.
    """
    return re.sub(r'\[.*?\]', '', text)


def _remove_parentheses(text):
    """
    Removes parentheses from the text.
    """
    return re.sub(r'\(.*?\)', '', text)


def _remove_urls(text):
    """
    Removes URLs from the text.
    """
    return re.sub(r'\b(?:http|ftp)s?://\S+', '', text)


def _remove_non_alphanumeric(text):
    """
    Removes non-alphanumeric characters from the text.
    """
    return re.sub(r'\W', ' ', text)


def _remove_numbers(text):
    """
    Removes numbers from the text.
    """
    return re.sub(r'\d+', '', text)


def get_basename(file_path):
    """
    Retrieves the basename of a file path.
    """
    return os.path.basename(file_path)


def remove_excessive_whitespace(text):
    """
    Removes excessive whitespace from the text.
    """
    return re.sub(r'\s+', ' ', text).strip()