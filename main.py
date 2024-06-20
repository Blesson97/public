import os
import tempfile
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.llm_chain import LLMChain
from langchain.prompt_template import PromptTemplate

from config import model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from questions import ask_question, QuestionContext


load_dotenv()


def generate_prompt_template(repo_name, github_url, conversation_history, question, numbered_documents, file_type_counts,
                             filenames):
    """
    Generates the prompt template for asking a question.

    Args:
        repo_name (str): The name of the repository.
        github_url (str): The URL of the GitHub repository.
        conversation_history (str): The conversation history.
        question (str): The question being asked.
        numbered_documents (int): The number of documents in the repository.
        file_type_counts (dict): The counts of each file type in the repository.
        filenames (list): The names of the files in the repository.

    Returns:
        PromptTemplate: The generated prompt template.
    """
    template = """
    Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}
    Instr:
    1. Answer based on context/docs.
    2. Focus on repo/code.
    3. Consider:
       a. Purpose/features - describe.
       b. Functions/code - provide details/samples.
       c. Setup/usage - give instructions.
    4. Unsure? Say "I am not sure".
    Answer:"
    """

    input_variables = ["repo_name", "github_url", "conversation_history", "question", "numbered_documents",
                       "file_type_counts", "filenames"]
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    return prompt


def clone_and_index_repository(github_url):
    """
    Clone and index the repository.

    Args:
        github_url (str): The URL of the GitHub repository.

    Returns:
        tuple: The index, documents, file type counts, and filenames.
    """
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            return index, documents, file_type_counts, filenames
        else:
            return None, None, None, None


def prompt_for_question(user_input, question_context, conversation_history):
    """
    Prompt for a question, generate an answer, and update conversation_history.

    Args:
        user_input (str): The user's input.
        question_context (QuestionContext): The question context object.
        conversation_history (str): The conversation history.

    Returns:
        str: The generated answer.
    """
    user_question = format_user_question(user_input)
    answer = ask_question(user_question, question_context)
    conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
    return answer


def ask_questions(repo_name, github_url, index, documents, file_type_counts, filenames, llm_chain,
                   conversation_history):
    """
    Ask questions about the repository and display the answers.

    Args:
        repo_name (str): The name of the repository.
        github_url (str): The URL of the GitHub repository.
        index (object): The index of the repository.
        documents (list): The list of documents in the repository.
        file_type_counts (dict): The counts of each file type in the repository.
        filenames (list): The names of the files in the repository.
        llm_chain (LLMChain): The LLMChain object.
        conversation_history (str): The conversation history.
    """
    question_context = QuestionContext(
        index,
        documents,
        llm_chain,
        model_name,
        repo_name,
        github_url,
        conversation_history,
        file_type_counts,
        filenames
    )

    while True:
        try:
            user_input = input(f"\nAsk a question about the repository (type 'exit()' to quit): ")
            if user_input.lower() == "exit()":
                break
            else:
                print('Thinking...')
                answer = prompt_for_question(user_input, question_context, conversation_history)
                print(f"\nANSWER\n{answer}\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            break


def main():
    """
    The main entry point of the program.
    """
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = os.path.basename(github_url)

    print("Cloning the repository...")
    index, documents, file_type_counts, filenames = clone_and_index_repository(github_url)

    if not index or not documents:
        print("No documents were found to index. Exiting.")
        return

    print("Repository cloned. Indexing files...")

    prompt = generate_prompt_template(repo_name, github_url, "", "", len(documents), file_type_counts, filenames)
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(api_key=load_dotenv("OPENAI_API_KEY"), temperature=0.2))

    conversation_history = ""
    ask_questions(repo_name, github_url, index, documents, file_type_counts, filenames, llm_chain, conversation_history)


if __name__ == "__main__":
    main()