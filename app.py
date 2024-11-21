import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
import re

class CustomMathSolver:
    @staticmethod
    def solve_math_problem(problem):
        try:
            problem = re.sub(r'[^0-9+\-*/().\s]', '', problem)
            result = eval(problem)
            return f"Solution: {problem} = {result}"
        except Exception as e:
            return f"Error solving math problem: {str(e)}"

st.set_page_config(page_title="Math Problem Solver")
st.title("Advanced Math Problem Solver")

with st.sidebar:
    groq_api_key = st.text_input(label="Groq API", type="password")
    model_selection = st.selectbox(
        "Select model",
        [
            'llama-3.1-70b-versatile', 
            'llama-3.1-8b-instant', 
            'gemma-7b-it',
            'llama-3.2-90b-vision-preview', 
            'llama3-70b-8192', 
            "mixtral-8x7b-32768"
        ]
    )

if not groq_api_key:
    st.info("Please add your Groq API key")
    st.stop()

model = ChatGroq(model=model_selection, groq_api_key=groq_api_key)
wikipedia = WikipediaAPIWrapper()

math_solver = CustomMathSolver()

tools = [
    Tool(
        name="Math Solver",
        func=math_solver.solve_math_problem,
        description="Solve mathematical expressions and basic arithmetic"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Search Wikipedia for additional information"
    )
]

assistant = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parse_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can help you solve mathematical problems. Enter a math expression or question."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

question = st.text_area("Enter your mathematical question or expression")

if st.button("Solve"):
    if question:
        with st.spinner("Calculating solution..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            try:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

                math_result = math_solver.solve_math_problem(question)
                
                if "Error" in math_result:
                    response = assistant.run(input=question, callbacks=[st_cb])
                else:
                    response = math_result

                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.write("Solution:")
                st.success(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a mathematical question")