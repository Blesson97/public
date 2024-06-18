# questions.py

from utils import format_documents
from file_processing import search_documents


class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history,
                 file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames


def ask_question(question, context: QuestionContext):
    """
    Process the given question and return the generated answer.

    Args:
        question (str): The question being asked.
        context (QuestionContext): The context containing relevant information.

    Returns:
        str: The generated answer.
    """
    relevant_docs = get_relevant_documents(question, context)
    question_context = generate_question_context(context)
    answer = generate_answer(question, question_context, context)
    return answer


def get_relevant_documents(question, context):
    """
    Retrieve relevant documents based on the given question and context.

    Args:
        question (str): The question being asked.
        context (QuestionContext): The context containing relevant information.

    Returns:
        list: List of relevant documents.
    """
    return search_documents(question, context.index, context.documents, n_results=5)


def generate_answer(question, question_context, context):
    """
    Generate an answer to the given question based on the question context.

    Args:
        question (str): The question being asked.
        question_context (str): The context of the question.
        context (QuestionContext): The context containing relevant information.

    Returns:
        str: The generated answer.
    """
    return context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context
    )


def generate_question_context(context):
    """
    Generate the question context based on the given context.

    Args:
        context (QuestionContext): The context containing relevant information.

    Returns:
        str: The generated question context.
    """
    formatted_docs = format_documents(context.documents)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{formatted_docs}"
    return question_context