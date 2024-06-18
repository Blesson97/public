# questions.py

from file_processing import search_documents


class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url):
        """
        A class representing the context of a question.

        Attributes:
        - index: The index of the context.
        - documents: The list of documents in the context.
        - llm_chain: The LLM chain used for generating answers.
        - model_name: The name of the model used for generating answers.
        - repo_name: The name of the GitHub repository.
        - github_url: The URL of the GitHub repository.
        """
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url


def ask_question(question: str, context: QuestionContext) -> str:
    """
    Process the given question and return the generated answer.

    Args:
        question: The question being asked.
        context: The context containing relevant information.

    Returns:
        str: The generated answer.
    """
    relevant_docs = _get_relevant_documents(question, context)
    question_context = _generate_question_context(context)
    answer = _generate_answer(question, question_context, context)
    return answer


def _get_relevant_documents(question: str, context: QuestionContext) -> list:
    """
    Retrieve relevant documents based on the given question and context.

    Args:
        question: The question being asked.
        context: The context containing relevant information.

    Returns:
        list: List of relevant documents.
    """
    return search_documents(question, context.index, context.documents, n_results=5)


def _generate_answer(question: str, question_context: str, context: QuestionContext) -> str:
    """
    Generate an answer to the given question based on the question context.

    Args:
        question: The question being asked.
        question_context: The context of the question.
        context: The context containing relevant information.

    Returns:
        str: The generated answer.
    """
    return context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context
    )


def _generate_question_context(context: QuestionContext) -> str:
    """
    Generate the question context based on the given context.

    Args:
        context: The context containing relevant information.

    Returns:
        str: The generated question context.
    """
    formatted_docs = _format_documents(context.documents)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{formatted_docs}"
    return question_context

def _format_documents(documents: list) -> str:
    """
    Format the documents into a readable string.

    Args:
        documents: The list of documents.

    Returns:
        str: The formatted documents.
    """
    formatted_docs = '\n'.join(documents)
    return formatted_docs