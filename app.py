import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="My Dashboard", layout="wide")

st.title("📊 Interactive Dashboard Example")

# Load sample data
df = px.data.gapminder()

# Sidebar filter
continent = st.sidebar.selectbox("Select Continent", df["continent"].unique())

# Filter data
filtered_df = df[df["continent"] == continent]

# Plot
fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                 color="country", size="pop",
                 hover_name="country", log_x=True,
                 title=f"Life Expectancy vs GDP per Capita ({continent})")

st.plotly_chart(fig, use_container_width=True)

