import streamlit as st
import re, time, ast
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# --- 1) Connect to Neo4j ---
graph = Neo4jGraph(
    url="neo4j://127.0.0.1:7687",
    username="neo4j",
    password="datvip01",
    database="neo4j",
    refresh_schema=False,
)

# --- 2) Setup LLM ---
llm = Ollama(model="gemma3:12b", temperature=0)

# --- 3) Schema & prompt for Cypher generation ---
schema = """Nodes:
  (:Subject {code, name, credits})
  (:Coordinator {name})
  (:TeachingPeriod {name})
  (:Topic {name})
  (:Outcome {description})
  (:AssumedKnowledge {name})
  (:Restriction {name})

Relationships:
  (:Subject)-[:COORDINATED_BY]->(:Coordinator)
  (:Subject)-[:OFFERED_IN]->(:TeachingPeriod)
  (:Subject)-[:COVERS]->(:Topic)
  (:Subject)-[:HAS_OUTCOME]->(:Outcome)
  (:Subject)-[:ASSUMES]->(:AssumedKnowledge)
  (:Subject)-[:RESTRICTED_BY]->(:Restriction)
  (:Subject)-[:REQUIRES]->(:Subject)

⚠️ STRICT SCHEMA — DO NOT DEVIATE ⚠️

Cypher Generation Rules:

1. Subject Code Format:
   - Subject codes always follow the format: 4 uppercase letters + 4 digits (e.g., MATH1234).
   - When returning results, always include distinct subject code and subject name only.
   - When returning results, do not use function EXISTS.
   - The only correct relationship about coordinator is MATCH (s:Subject)-[:COORDINATED_BY]->(c:Coordinator)
   - Subject name can be used to filter (e.g., Predictive Analytics)

2. Teaching Period Filtering:
   - Teaching periods are nodes: (:TeachingPeriod {name}).
   - When the question mentioned enroll or register the teaching period should be filtered.
   - To filter by season, use case-insensitive matching:
     ✅ Correct: WHERE toLower(tp.name) STARTS WITH "autumn 2023"
     ❌ Incorrect: WHERE tp.name = "Autumn 2023"
   - DO NOT USE NOT clause in teaching period filtering.

3. Exclusive Offering Logic:
   - To find subjects offered *only* in a specific season:
     ✅ Use NOT EXISTS with curly braces:
     ```
     MATCH (s:Subject)-[:OFFERED_IN]->(tp:TeachingPeriod)
     WHERE toLower(tp.name) STARTS WITH "autumn"
     AND NOT EXISTS {
       MATCH (s)-[:OFFERED_IN]->(tp2:TeachingPeriod)
       WHERE NOT toLower(tp2.name) STARTS WITH "autumn"
     }
     RETURN s.code, s.name
     ```
     ❌ Do NOT use EXISTS(...) or NOT EXISTS(...)

4. Subquery Syntax:
   - Always use `{}` blocks for subqueries.
   - NEVER use parentheses after EXISTS or NOT EXISTS.

5. NOT Usage:
   ✅ Correct: MATCH ... WHERE NOT toLower(tp.name) STARTS WITH "spring" AND NOT EXISTS { MATCH ... } RETURN ...
   ❌ Incorrect: WHERE tp.name NOT STARTS WITH "spring"

6. Prerequisite Relationship:
- Only use: (subject:Subject)-[:REQUIRES]->(preqsub:Subject)
- ❌ Do NOT use REQUIREES, REQUIRIES, REQUIRED_BY
- Use NOT EXISTS to see if a subject have no prerequisite or not. Example: WHERE NOT EXISTS {MATCH (s)-[:REQUIRES]->(preqsub:Subject)}

7. Discipline/area intent:
- If the question asks for subjects in a discipline (e.g., "Statistics"), search semantic fields (Outcome.description and/or Topic.name),
not code prefixes, unless explicitly requested.
  Example:
    MATCH ... WHERE toLower(o.description) CONTAINS "..." RETURN ...
    or using
    MATCH ... WHERE toLower(T.name) CONTAINS "..." RETURN ...

8. Completed Subjects Logic:
- Always use [] instead of () in where clause when using WHERE NOT s.code IN [...]
- When the question mentions "after finishing/completed" a list of subjects:
    * Exclude those completed subjects:
    WHERE NOT s.code IN [...]
    * Ensure prerequisites are satisfied:
    AND NOT EXISTS {
        MATCH (s)-[:REQUIRES]->(p:Subject)
        WHERE NOT p.code IN [...]
    }
- This guarantees that all prerequisites of s are contained in the completed set.
- ❌ Do NOT write queries that only check `s.code IN [...]` without the NOT EXISTS prerequisite logic.

9. Restriction
- If the question asks for subjects in a restriction (e.g., "point"), search semantic fields (Restriction.name),
not code prefixes, unless explicitly requested.
- Must return restriction name.
  Example:
    MATCH ... WHERE toLower(o.name) CONTAINS "..." RETURN ...

10. Skill/assessment intent (e.g., "presentations", "machine", "python", "learning", "programming", "project"):
- Never combine different graph patterns with OR (e.g., outcomes or topics) in a single branch.
- Always search in both Outcome.description and Topic.name (case-insensitive, partial match).
- To combine results from different relationships or node types (e.g., HAS_OUTCOME and COVERS), use two independent branches joined by UNION.
- Each branch of the UNION must have its own RETURN clause and own WHERE clause, its own RETURN DISTINCT s.code, s.name.
- The RETURN clauses must have identical columns (e.g., s.code AS subjectCode, s.name AS subjectName).
- The search term must be treated as a whole phrase from the user query (e.g., "machine learning"), not split into individual words unless explicitly asked.
- Always apply DISTINCT in each RETURN to remove duplicates.
    Example the correct cypher:
    
    MATCH (s:Subject)-[:HAS_OUTCOME]->(o:Outcome)
    WHERE toLower(o.description) CONTAINS "python"
    RETURN DISTINCT s.code, s.name # must include
    UNION
    MATCH (s:Subject)-[:COVERS]->(t:Topic)
    WHERE toLower(t.name) CONTAINS "python"
    RETURN DISTINCT s.code, s.name

11. Variable Definition Order:
   - A variable (like `tp`) must be introduced in a MATCH clause before it is referenced in WHERE.
   - Always place MATCH before any WHERE filters that reference its variables."""

cypher_prompt = PromptTemplate.from_template(
    """You are a Cypher expert. Using ONLY this schema:

{schema}

Write a SINGLE read-only Cypher query that answers:
{question}
"""
)

cypher_only = LLMChain(
    llm=llm,
    prompt=cypher_prompt.partial(schema=schema)
)

# --- 4) Clean cypher utility ---
def clean_cypher(cypher_text: str) -> str:
    cleaned = re.sub(r"```[a-zA-Z]*", "", cypher_text)
    cleaned = cleaned.replace("cypher\n", "")
    cleaned = re.sub(r"\bREQUIREES\b", "REQUIRES", cleaned)
    cleaned = re.sub(r"\bREQUIRIES\b", "REQUIRES", cleaned)
    return cleaned.replace("```", "").strip()

# --- 5) Full pipeline: question -> Cypher -> rows -> NL answer ---
def answer_question(question: str) -> str:
    start = time.time()

    # Generate Cypher
    cypher = cypher_only.invoke({"question": question})["text"].strip()
    cypher_query = clean_cypher(cypher)

    # Run Cypher query
    try:
        rows = graph.query(cypher_query)
    except Exception as e:
        return f"⚠️ Error running query: {e}"

    # Parse result into something clean
    parsed = rows if rows else []

    # Prompt to convert into NL
    prompt = f"""
Question: {question}
Cypher Answer: {parsed}

Ignore the subject name and convert the Cypher Answer into a short and direct natural answer that answers the question clearly.
    """

    # Convert using LLM
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = f"[LLM Error: {e}]"

    elapsed = round(time.time() - start, 2)
    return f"{response}\n\n⏱️ Answered in {elapsed} sec"

st.set_page_config(page_title="Academic Advisor Assistant", page_icon="🎓")

st.title("🎓 Academic Advisor Assistant")
st.write("Chat with the assistant about subjects, prerequisites, or outcomes.")

# keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input box at bottom
if user_input := st.chat_input("Ask your question here..."):
    # store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            nl_answer = answer_question(user_input)
            st.markdown(nl_answer)
    st.session_state.messages.append({"role": "assistant", "content": nl_answer})
