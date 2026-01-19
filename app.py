import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from backend import RAGOrchestrator, initialize_system

# 1. Page Config
st.set_page_config(page_title="Enterprise Governance Chatbot", page_icon="🔒", layout="wide")
st.title("🔒 Enterprise RAG Chatbot")

# 2. Get API Key (Smart Check)
api_key = None
# Check if key is in Secrets (Best Practice)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
# If not in secrets, ask in Sidebar
else:
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Google API Key", type="password")

# Stop if no key found anywhere
if not api_key:
    st.warning("⚠️ Google API Key missing. Please add it to Streamlit Secrets or the sidebar.")
    st.stop()

# 3. Sidebar - Authentication
with st.sidebar:
    st.divider()
    st.header("🔐 Authentication")
    
    if "auth_status" not in st.session_state:
        st.session_state.auth_status = False
        st.session_state.user_session = "session_user_1"

    # Login Form
    if not st.session_state.auth_status:
        st.info("Log in to access personal data.")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if "orchestrator" in st.session_state:
                result = st.session_state.orchestrator.login(email, password, st.session_state.user_session)
                if result.verified:
                    st.session_state.auth_status = True
                    st.session_state.user_email = email
                    st.success("✅ Logged in!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")
            else:
                st.warning("System initializing... try again in 5 seconds.")
    
    # Logout Button
    else:
        st.success(f"👤 Logged in as: {st.session_state.user_email}")
        if st.button("Logout"):
            st.session_state.auth_status = False
            if "orchestrator" in st.session_state:
                st.session_state.orchestrator.logout(st.session_state.user_session)
            del st.session_state.user_email
            st.rerun()

# 4. Initialize System (Cached in Session)
if "orchestrator" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        try:
            data_df, creds_df, vector_store = initialize_system()
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
            st.session_state.orchestrator = RAGOrchestrator(data_df, creds_df, vector_store, llm)
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I can answer general questions. Log in for personal data."}]
        except FileNotFoundError:
            st.error("🚨 Error: CSV files not found. Please upload `RAGbot_finance_enriched.csv` and `dummy_employee_credentials.csv` to GitHub.")
            st.stop()
        except Exception as e:
            st.error(f"🚨 System Error: {e}")
            st.stop()

# 5. Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about policies or your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # This is where "Thinking..." happens. If it fails, we catch the error.
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.orchestrator.process_query(prompt, st.session_state.user_session)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error processing query: {e}")
