import streamlit as st

st.set_page_config(
    page_title="Software Effort Estimator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/1_Estimator.py", title="COCOMO-81 Estimator", icon="📊", default=True),
    st.Page("pages/2_GitHub_Model.py", title="GitHub Model", icon="📈"),
    st.Page("pages/3_Your_Data.py", title="Your Data", icon="🏢"),
])
pg.run()
