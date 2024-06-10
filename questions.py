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
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)

    numbered_documents = format_documents(relevant_docs)
    question_context = generate_question_context(context)

    answer_with_sources = context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context
    )
    return answer_with_sources


def generate_question_context(context: QuestionContext):
    numbered_documents = format_documents(context.documents)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}"
    return question_context


# End of code