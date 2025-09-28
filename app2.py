import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama

# Initialize LLM
llm = Ollama(model="gemma3:12b", temperature=0)

# Connect to Neo4j
graph = Neo4jGraph(
    url="neo4j://127.0.0.1:7687",
    username="neo4j",
    password="datvip01",
    database="neo4j",
    refresh_schema=False,
)

st.set_page_config(page_title="Academic Advisor Assistant", page_icon="📘", layout="wide")
st.title("📘 Academic Advisor Assistant")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # store {"role": "user"/"assistant", "content": str}

# --- Sidebar: options ---
with st.sidebar:
    st.header("⚙️ Options")
    show_cypher = st.checkbox("Show Generated Cypher", value=False)
    show_raw_result = st.checkbox("Show Raw Results", value=False)
    clear = st.button("Clear Conversation")
    if clear:
        st.session_state.messages = []

# --- Dropdown actions ---
action = st.selectbox("Select action:", [
    "Which subjects are offered in",
    "Which subjects cover/require",
    "Which subjects list as prerequisite",
    "Does subject have prerequisites",
    "After completing subjects, which can I take in",
    "Which subjects have restriction",
    "Who coordinates subject"
])

placeholders = {
    "Which subjects are offered in": "Enter Teaching Period (e.g., Autumn 2025)",
    "Which subjects cover/require": "Enter skill, topic, or outcome (e.g., Python, Presentation)",
    "Which subjects list as prerequisite": "Enter prerequisite subject code (e.g., MATH7016)",
    "Does subject have prerequisites": "Enter subject code (e.g., COMP7023)",
    "After completing subjects, which can I take in": "Enter Teaching Period (e.g., Spring 2024)",
    "Which subjects have restriction": "Enter restriction keyword (e.g., credit point)",
    "Who coordinates subject": "Enter subject code (e.g., COMP7001)"
}

filter_val = st.text_input(placeholders[action])

completed_subjects = []
if "After completing" in action:
    completed_subjects = st.text_area(
        "Enter completed subject codes (comma-separated):",
        placeholder="MATH7002, COMP7006, COMP7023"
    ).split(",")

# --- Submit button ---
if st.button("💡 Ask"):
    # Build Cypher query (same as your logic before)
    cypher = "MATCH (s:Subject) RETURN s.code, s.name"
    if action == "Which subjects are offered in":
        cypher = f"""
        MATCH (s:Subject)-[:OFFERED_IN]->(tp:TeachingPeriod)
        WHERE toLower(tp.name) CONTAINS '{filter_val.lower()}'
        RETURN DISTINCT s.code, s.name
        """
    elif action == "Which subjects cover/require":
        cypher = f"""
        MATCH (s:Subject)-[:HAS_OUTCOME]->(o:Outcome)
        WHERE toLower(o.description) CONTAINS '{filter_val.lower()}'
        RETURN DISTINCT s.code, s.name
        UNION
        MATCH (s:Subject)-[:COVERS]->(t:Topic)
        WHERE toLower(t.name) CONTAINS '{filter_val.lower()}'
        RETURN DISTINCT s.code, s.name
        """
    elif action == "Which subjects list as prerequisite":
        cypher = f"""
        MATCH (s:Subject)-[:REQUIRES]->(p:Subject {{code:'{filter_val}'}})
        RETURN DISTINCT s.code, s.name
        """
    elif action == "Does subject have prerequisites":
        cypher = f"""
        MATCH (s:Subject {{code:'{filter_val}'}})
        OPTIONAL MATCH (s)-[:REQUIRES]->(p:Subject)
        RETURN s.code, s.name, collect(p.code) AS prerequisites
        """
    elif action == "After completing subjects, which can I take in":
        completed_str = [c.strip() for c in completed_subjects if c.strip()]
        cypher = f"""
        MATCH (s:Subject)-[:OFFERED_IN]->(tp:TeachingPeriod)
        WHERE NOT s.code IN {completed_str}
        AND NOT EXISTS {{
          MATCH (s)-[:REQUIRES]->(p:Subject)
          WHERE NOT p.code IN {completed_str}
        }}
        AND toLower(tp.name) CONTAINS '{filter_val.lower()}'
        RETURN DISTINCT s.code, s.name
        """
    elif action == "Which subjects have restriction":
        cypher = f"""
        MATCH (s:Subject)-[:RESTRICTED_BY]->(r:Restriction)
        WHERE toLower(r.name) CONTAINS '{filter_val.lower()}'
        RETURN DISTINCT s.code, s.name, r.name
        """
    elif action == "Who coordinates subject":
        cypher = f"""
        MATCH (s:Subject {{code:'{filter_val}'}})-[:COORDINATED_BY]->(c:Coordinator)
        RETURN s.code, s.name, c.name AS coordinator
        """

    result = graph.query(cypher)
    if result:
        prompt = f"Question: {action} {filter_val}\nResult: {result}\n\nConvert this result into a clear short natural language answer."
        nl_answer = llm.invoke(prompt)
    else:
        nl_answer = "No subjects found that match your criteria."

    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": f"{action} {filter_val}"})
    st.session_state.messages.append({"role": "assistant", "content": nl_answer})

    # Optionally show debug info
    if show_cypher:
        st.write("🔎 Cypher used:")
        st.code(cypher, language="cypher")
    if show_raw_result:
        st.write("📋 Raw results:", result)

# --- Chat history display ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
