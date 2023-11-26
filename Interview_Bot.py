import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM
llm = OpenAI()

# Initialize the prompt template
interview_prompt = PromptTemplate(
    input_variables=["job_role", "question"],
    template="Based on the candidate's previous answer for a {job_role} role, what would be your next question? Previous answer: {question}"
)


# Function to ask a question
def ask_question(job_role, answer):
    filled_prompt = interview_prompt.format(job_role=job_role, question=answer)
    response = llm(filled_prompt)
    return response


# Function to conduct the interview
def conduct_interview(job_role):
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.answers = []

    questions = ["Can you tell me a little about yourself?", "", "", ""]  # Initial question and placeholders for others

    if st.session_state.question_index < 4:
        question = questions[st.session_state.question_index]
        user_input = st.text_input(f"Question {st.session_state.question_index + 1}: {question}",
                                   key=f'Question_{st.session_state.question_index + 1}')

        if user_input:
            st.session_state.answers.append(user_input)
            st.session_state.question_index += 1
            if st.session_state.question_index < 4:
                # Generate next question based on the response
                questions[st.session_state.question_index] = ask_question(job_role, user_input)

    # Providing a summary after all questions have been answered
    if st.session_state.question_index == 4:
        st.write("Interview completed. Here is a summary of your responses:")
        for idx, answer in enumerate(st.session_state.answers, 1):
            st.write(f"Q{idx}: {answer}")
        # Example summary
        summary = "Thank you for your responses. You've demonstrated a good understanding of the role."
        st.write(summary)

        # Clear button
        if st.button('Clear'):
            st.session_state.clear()
            st.experimental_rerun()


# Basic Layout of the Site
with st.sidebar:
    st.sidebar.title('Interview App')

# Job Role Selection
if 'selected_job' not in st.session_state:
    st.session_state.selected_job = None

job_buttons = ["Data Science", "Full Stack Developer", "BackEnd Dev", "Management"]
for job in job_buttons:
    if st.button(job):
        st.session_state.selected_job = job

# Conduct Interview
if st.session_state.selected_job:
    st.header(f"Interview for {st.session_state.selected_job} Role")
    conduct_interview(st.session_state.selected_job)
else:
    st.header("Please select a job role to begin the interview.")
