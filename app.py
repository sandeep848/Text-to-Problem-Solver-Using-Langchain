"""Verified symbolic mathematics interface."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from solver import solve_problem

load_dotenv()
st.set_page_config(page_title="Verified Math Problem Solver")
st.title("Verified Math Problem Solver")
st.caption("SymPy computes and verifies the answer; the LLM only explains it.")

problem = st.text_input(
    "Expression or equation",
    placeholder="Examples: 2x + 5 = 17, (3 + 4)^2 / 7",
)
api_key = st.sidebar.text_input(
    "Optional Groq API key", type="password", value=os.getenv("GROQ_API_KEY", "")
)

if st.button("Solve and verify", type="primary", disabled=not problem):
    try:
        result = solve_problem(problem)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.metric("Verified answer", result.answer)
    st.code(result.verification)

    if api_key:
        llm = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0,
        )
        explanation = llm.invoke(
            f"""Explain this verified symbolic solution for a student.
Do not change the answer and do not invent extra assumptions.
Problem: {result.normalized_problem}
Answer: {result.answer}
Verification: {result.verification}"""
        ).content
        st.subheader("Explanation")
        st.write(explanation)
    else:
        st.info("Add a Groq key only if you want a natural-language explanation.")
