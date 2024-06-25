# questions.py

from typing import List
from file_processing import search_documents
from llm_chain import LLMChain
from model import Model


class QuestionContext:
    """
    A class representing the context of a question.
    """

    def __init__(self, index: int, documents: List[str], llm_chain: LLMChain, model: Model, repo_name: str,
                 github_url: str):
        """
        Initialize the QuestionContext object.

        Parameters:
            index (int): The index of the context.
            documents (List[str]): The list of documents in the context.
            llm_chain (LLMChain): The LLM chain used for generating answers.
            model (Model): The model used for generating answers.
            repo_name (str): The name of the GitHub repository.
            github_url (str): The URL of the GitHub repository.
        """
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model = model
        self.repo_name = repo_name
        self.github_url = github_url


def generate_answer(llm_chain: LLMChain, model: Model, question_context: str) -> str:
    """
    Generate the answer using the LLM chain and model.

    Parameters:
        llm_chain (LLMChain): The LLM chain used for generating answers.
        model (Model): The model used for generating answers.
        question_context (str): The question context.

    Returns:
        str: The generated answer.
    """
    return llm_chain.run(model=model, context=question_context)


def search_relevant_documents(question: str, context: QuestionContext) -> List[str]:
    """
    Search for relevant documents based on the question.

    Parameters:
        question (str): The question being asked.
        context (QuestionContext): The context containing relevant information.

    Returns:
        List[str]: The relevant documents.
    """
    return search_documents(question, context.index, context.documents, n_results=5)


def generate_question_context(context: QuestionContext) -> str:
    """
    Generate the question context string.

    Parameters:
        context (QuestionContext): The context containing relevant information.

    Returns:
        str: The question context string.
    """
    repo_name = context.repo_name
    github_url = context.github_url
    relevant_docs = context.documents
    return f"This question is about the GitHub repository '{repo_name}' available at {github_url}. The most relevant documents are:\n\n{'\n'.join(relevant_docs)}"


def process_question(question: str, context: QuestionContext) -> str:
    """
    Process the given question and return the generated answer.

    Parameters:
        question (str): The question being asked.
        context (QuestionContext): The context containing relevant information.

    Returns:
        str: The generated answer.
    """
    relevant_docs = search_relevant_documents(question, context)
    question_context = generate_question_context(context)
    answer = generate_answer(context.llm_chain, context.model, question_context)
    return answer

# End of refactored code.
#modification
