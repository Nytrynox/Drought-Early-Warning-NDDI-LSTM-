import streamlit as st


def get_session_state():
    if "_initialized" not in st.session_state:
        st.session_state._initialized = True
    return st.session_state
