import streamlit as st
from RAG import agent_executor

st.set_page_config(page_title="Agri Assistant", page_icon="🌾", layout="centered")

st.title("🌾 Agricultural Assistant")
st.markdown(
    """
    Ask your farming and agriculture-related questions.  
    This assistant uses internal documents and web search to provide detailed, structured answers.
    """
)

query = st.text_area("Enter your question here:", height=100)


if st.button("Get Answer") and query.strip():
    with st.spinner("Thinking..."):
        result = agent_executor.invoke({"input": query})
    st.markdown("### Final Answer")
    st.markdown(result.get("output", "No answer generated."))

else:
    st.info("Please enter a question to get started.")
