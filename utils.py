# utils.py

import re
import os

from nltk.tokenize import word_tokenize


def clean_and_tokenize(text):
    """
    Cleans and tokenizes the text.
    Removes HTML tags, square brackets, parentheses, URLs, non-alphanumeric characters,
    numbers, and converts the text to lowercase.
    """
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses
    text = re.sub(r'\b(?:http|ftp)s?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()

    return word_tokenize(text)

def format_documents(documents):
    """
    Formats the list of documents.
    Returns a numbered list with the basename of the source file and the page content of each document.
    """
    numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content}" for i, doc in enumerate(documents)])
    return numbered_docs

def format_user_question(question):
    """
    Formats the user's question.
    Removes excessive whitespace.
    """
    question = re.sub(r'\s+', ' ', question).strip()
    return question
