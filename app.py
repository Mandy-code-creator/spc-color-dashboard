import streamlit as st
import pandas as pd

st.title("SPC – Test kết nối Google Sheet")

url = "https://docs.google.com/spreadsheets/d/1lqsLKSoDTbtvAsHzJaEri8tPo5pA3vqJ__LVHp2R534/gviz/tq?tqx=out:csv"

df = pd.read_csv(url)

st.success("Đọc Google Sheet thành công!")
st.write(df.head())
