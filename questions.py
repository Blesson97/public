# questions.py

from typing import List

from file_processing import search_documents
from llm_chain import LLMChain
from model import Model


class QuestionContext:
    """
    A class representing the context of a question.
    """

    def __init__(
        self,
        index: int,
        documents: List[str],
        llm_chain: LLMChain,
        model: Model,
        repo_name: str,
        github_url: str,
    ):
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


def ask_question(question: str, context: QuestionContext) -> str:
    """
    Process the given question and return the generated answer.

    Parameters:
        question (str): The question being asked.
        context (QuestionContext): The context containing relevant information.

    Returns:
        str: The generated answer.
    """
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)
    question_context = generate_question_context(context.repo_name, context.github_url, relevant_docs)
    answer = generate_answer(context.llm_chain, context.model, question, question_context)
    return answer


def generate_question_context(repo_name: str, github_url: str, relevant_docs: List[str]) -> str:
    """
    Generate the question context string.

    Parameters:
        repo_name (str): The name of the GitHub repository.
        github_url (str): The URL of the GitHub repository.
        relevant_docs (List[str]): The list of relevant documents.

    Returns:
        str: The question context string.
    """
    return f"This question is about the GitHub repository '{repo_name}' available at {github_url}. The most relevant documents are:\n\n{'\n'.join(relevant_docs)}"


def generate_answer(llm_chain: LLMChain, model: Model, question: str, question_context: str) -> str:
    """
    Generate the answer using the LLM chain and model.

    Parameters:
        llm_chain (LLMChain): The LLM chain.
        model (Model): The model.
        question (str): The question.
        question_context (str): The question context.

    Returns:
        str: The generated answer.
    """
    return llm_chain.run(model=model, question=question, context=question_context)