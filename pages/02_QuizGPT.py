import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema import BaseOutputParser

class JsonOutputParser(BaseOutputParser):  
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from the model's response.")
            return {}

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs, difficulty):
    # Adjust the context based on difficulty
    num_questions = 5 if difficulty == "Easy" else 10 if difficulty == "Medium" else 15
    context_length = difficulty_levels[difficulty]
    return "\n\n".join(document.page_content[:context_length] for document in docs[:num_questions])

difficulty_levels = {
    "Easy": 300,
    "Medium": 600,
    "Hard": 900,
}

@st.cache_data(show_spinner="Loading file...")
def split_file(file, difficulty):
    file_content = file.read()
    chunk_size = difficulty_levels[difficulty]
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def get_prompt_template(difficulty):
    context_length = difficulty_levels[difficulty]
    return ChatPromptTemplate.from_messages([
        (
                "system",
                f"""
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context (up to {context_length} characters), make questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {{context}}
""",
            )
    ])

formatting_prompt = ChatPromptTemplate.from_messages([
    (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
])

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Making quiz....")
def run_quiz_chain(_docs, difficulty, topic):
    if not _docs:
        st.error("No documents available to generate questions.")
        return {'questions': []}
        
    questions_prompt = get_prompt_template(difficulty)
    questions_chain = {"context": (lambda docs: format_docs(docs, difficulty))} | questions_prompt | llm
    formatting_chain = formatting_prompt | llm
    chain = {"context": questions_chain} | formatting_chain | output_parser
    response = chain.invoke(_docs)
    
    if not response or 'questions' not in response:
        st.error("No questions were generated. Please check the input data or configuration.")
        return {'questions': []}
    
    return response
    

@st.cache_data(show_spinner="Searching Wikipedia....")
def wiki_search(term, difficulty):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term) 
    return docs[:difficulty_levels[difficulty] // 100]

with st.sidebar:
    docs = None
    difficulty = st.selectbox("Select difficulty", ["Easy", "Medium", "Hard"])
    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article",))

    if choice == "File":
        file = st.file_uploader("Upload a .docx , .txt or .pdf file", type=["pdf", "txt", "docx"])
        if file:
            docs = split_file(file, difficulty)
    else:
        topic = st.text_input("Search Wikipedia....")
        if topic:
            docs = wiki_search(topic, difficulty)

if not docs:
    st.markdown("""
        Welcome to QuizGPT.
                
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
        Get started by uploading a file or searching on Wikipedia in the sidebar.
    """)
else:
    response = run_quiz_chain(docs, difficulty, topic if topic else file.name)
    # 'questions' 키에 대한 확인은 run_quiz_chain 내부에서 처리
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("Select an option", [answer["answer"] for answer in question["answers"]])
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()
