# Verified Math Problem Solver

A hybrid mathematics application in which **SymPy performs the calculation and verification** while an optional language model explains the already-verified result.

This is intentionally different from a general chatbot: the LLM is not trusted to calculate the final answer.

## Logic

~~~mermaid
flowchart LR
    A["Equation or expression"] --> B["Restricted parser"]
    B --> C["SymPy solver"]
    C --> D["Residual verification"]
    D --> E["Verified answer"]
    E --> F["Optional LLM explanation"]
~~~

## Supported tasks

- arithmetic and simplification
- single- and multi-variable symbolic equations
- implicit multiplication such as `2x`
- exponent notation with `^`
- substitution-based residual checks

## Run

~~~bash
git clone https://github.com/sandeep848/langchain-math-problem-solver.git
cd langchain-math-problem-solver
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

The Groq key is optional because solving and verification run locally.

## Test

~~~bash
pytest
~~~

## Design boundary

Natural-language word problems still require an interpretation layer. The repository deliberately keeps that separate from symbolic computation so that a fluent explanation cannot silently replace mathematical verification.
