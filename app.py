import streamlit as st

st.error("🚨 YOU ARE RUNNING: streamlit_test_final.py")

st.set_page_config(
    page_title="STREAMLIT FILE TEST",
    layout="wide"
)

st.markdown(
    """
    <div style="padding:30px;border:4px solid red;">
        <h1 style="color:red;">
            🚨 STREAMLIT FILE TEST – MUST BE VISIBLE
        </h1>

        <h2 style="color:blue;">
            ⏱ 2024-07-03 → 2024-11-18 | n = 42 batches
        </h2>

        <h3 style="color:green;">
            Year: 2024 | Month: All
        </h3>

        <p style="font-size:20px;">
            If you see THIS SCREEN, you are running the CORRECT file.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("📄 Python file path:")
st.code(__file__)

st.success("✅ TEST FINISHED – NOTHING SHOULD BE MISSING")
