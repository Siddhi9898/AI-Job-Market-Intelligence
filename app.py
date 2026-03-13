import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import Counter
import PyPDF2

plt.style.use("seaborn-v0_8")
# Page config
st.set_page_config(
    page_title="AI Job Market Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Job Market Intelligence Platform")
st.divider()
st.markdown(
    """
Analyze AI & Data Science job trends, salary insights, and skill demand  
to understand the evolving job market.
"""
)

# Load data
df = pd.read_csv("data/jobs.csv")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home","Market Analysis","Skill Demand","Role Skill Analyzer",
     "Salary Analysis","Salary Predictor","Skill Gap Analyzer",
     "Skill Roadmap","Resume Skill Analyzer","Job Recommender","Insights","AI Job Trend Prediction"]
)

# Sidebar filters (apply globally)
role_filter = st.sidebar.multiselect(
    "Filter by Job Role", options=df["job_title"].unique(), default=df["job_title"].unique()
)
country_filter = st.sidebar.multiselect(
    "Filter by Country", options=df["location"].unique(), default=df["location"].unique()
)
filtered_df = df[df["job_title"].isin(role_filter) & df["location"].isin(country_filter)]

# --- HOME ---
if menu == "Home":
    st.header("Project Overview")
    st.write("""
    This platform analyzes job market trends in Data Science and AI.
    
    Features:
    - Job role demand
    - Skill demand analysis
    - Salary insights
    - Market intelligence
    """)

# --- MARKET ANALYSIS ---
elif menu == "Market Analysis":
    st.header("Job Market Overview")
    st.divider()
    st.caption("Real-time insights from AI & Data Science job postings")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Job Posts", len(filtered_df))
    col2.metric("Unique Job Roles", filtered_df["job_title"].nunique())
    col3.metric("Unique Companies", filtered_df["company"].nunique())
    col4.metric("Countries", filtered_df["location"].nunique())

    st.divider()

    # Top Skills Chart
    st.subheader("Top Skills in Demand")
    top_skills = filtered_df["skills"].value_counts().head(10)
    st.bar_chart(top_skills, use_container_width=True)

    st.divider()

    # Top Job Roles Chart
    st.subheader("Most Common Job Roles")
    fig, ax = plt.subplots()
    sns.countplot(
        y=filtered_df["job_title"],
        order=filtered_df["job_title"].value_counts().index[:10],
        ax=ax
    )
    st.pyplot(fig, use_container_width=True)

# --- SKILL DEMAND ---
elif menu == "Skill Demand":
    st.header("Top Skills in Job Market")
    st.caption("Insights based on all job postings")

    all_skills = []
    for skills in filtered_df["skills"].dropna():
        skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")
        all_skills.extend([s.strip() for s in skill_list])

    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(skill_counts.most_common(10), columns=["Skill","Count"])
    fig = px.bar(
        top_skills,
        x="Count",
        y="Skill",
        orientation="h",
        title="Top Skills in Demand"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- ROLE SKILL ANALYZER ---
elif menu == "Role Skill Analyzer":
    st.header("Skills Required for Each Job Role")
    role = st.selectbox("Select Job Role", filtered_df["job_title"].dropna().unique())
    role_data = filtered_df[filtered_df["job_title"] == role]

    all_skills = []
    for skills in role_data["skills"].dropna():
        skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")
        all_skills.extend([s.strip().lower() for s in skill_list])

    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(skill_counts.most_common(10), columns=["Skill","Count"])
    import plotly.express as px

    fig = px.bar(
        top_skills,
        x="Count",
        y="Skill",
        orientation="h",
        title="Top Skills in Demand"
    )

    st.plotly_chart(fig, use_container_width=True)
# SALARY ANALYSIS
elif menu == "Salary Predictor":

    st.header("Salary Predictor")

    if "salary_numeric" not in df.columns:
        st.write("Salary data not available")
    else:

        st.write("Enter experience level to estimate salary")

        experience = st.slider("Years of Experience", 0, 15, 1)

        X = np.array(range(len(df["salary_numeric"].dropna()))).reshape(-1,1)
        y = df["salary_numeric"].dropna().values

        model = LinearRegression()
        model.fit(X, y)

        prediction = model.predict([[experience]])

        st.subheader("Estimated Salary")

        st.write(int(prediction[0]))

# SKILL GAP ANALYZER
elif menu == "Skill Gap Analyzer":

    st.header("Skill Gap Analyzer")

    st.write("Enter your current skills (comma separated)")

    user_input = st.text_input("Your Skills")

    if user_input:

        user_skills = [skill.strip().lower() for skill in user_input.split(",")]

        all_skills = []

        for skills in df["skills"].dropna():
            skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")

            for s in skill_list:
                all_skills.append(s.strip().lower())

        skill_counts = Counter(all_skills)

        top_market_skills = [skill for skill, count in skill_counts.most_common(15)]

        missing_skills = [skill for skill in top_market_skills if skill not in user_skills]

        st.subheader("Recommended Skills to Learn")

        for skill in missing_skills[:10]:
            st.write(skill)

# SKILL ROADMAP
elif menu == "Skill Roadmap":

    st.header("AI Skill Roadmap Generator")

    role = st.selectbox("Choose Target Role", df["job_title"].dropna().unique())

    user_input = st.text_input("Enter Your Current Skills (comma separated)")

    if user_input:

        user_skills = [s.strip().lower() for s in user_input.split(",")]

        role_data = df[df["job_title"] == role]

        all_skills = []

        for skills in role_data["skills"].dropna():
            skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")

            for s in skill_list:
                all_skills.append(s.strip().lower())

        from collections import Counter
        skill_counts = Counter(all_skills)

        market_skills = [skill for skill,count in skill_counts.most_common(20)]

        missing_skills = [skill for skill in market_skills if skill not in user_skills]

        st.subheader("Recommended Learning Roadmap")

        step = 1
        for skill in missing_skills[:10]:
            st.write(f"Step {step}: Learn {skill}")
            step += 1
# INSIGHTS
elif menu == "Insights":

    st.header("Market Insights")

    top_role = df["job_title"].value_counts().idxmax()

    st.write(f"Most common job role: {top_role}")

    st.write("Top skills demanded:")

    all_skills = []

    for skills in df["skills"].dropna():
        skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")
        for s in skill_list:
            all_skills.append(s.strip())

    skill_counts = Counter(all_skills)

    for skill, count in skill_counts.most_common(5):
        st.write(f"{skill} : {count} job postings")

elif menu == "Resume Skill Analyzer":

    import PyPDF2

    st.header("Resume Skill Analyzer")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file is not None:

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""
        for page in reader.pages:
           page_text = page.extract_text()
           if page_text:
                text += page_text

        text = text.lower()

        all_skills = []

        for skills in df["skills"].dropna():
            skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")
            for s in skill_list:
                all_skills.append(s.strip().lower())

        market_skills = list(set(all_skills))

        found_skills = []

        for skill in market_skills:
            if skill.lower() in text:
                found_skills.append(skill)

        if len(found_skills) == 0:
            st.write("No skills detected from dataset.")
        for skill in found_skills[:15]:
            st.success(skill)
elif menu == "Job Recommender":

    st.header("AI Job Recommendation")

    user_input = st.text_input("Enter your skills (comma separated)")

    if user_input:

        user_skills = [s.strip().lower() for s in user_input.split(",")]

        role_scores = {}

        for role in df["job_title"].unique():

            role_data = df[df["job_title"] == role]

            role_skills = []

            for skills in role_data["skills"].dropna():
                skill_list = str(skills).replace("[","").replace("]","").replace("'","").split(",")

                for s in skill_list:
                    role_skills.append(s.strip().lower())

            score = len(set(user_skills) & set(role_skills))

            role_scores[role] = score

        recommended = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        st.subheader("Recommended Roles")

        for role, score in recommended:
            st.write(f"{role} (Match Score: {score})")

elif menu == "AI Job Trend Prediction":

    st.header("AI Job Demand Prediction")

    job_counts = df["job_title"].value_counts().head(5)

    X = np.arange(len(job_counts)).reshape(-1,1)
    y = job_counts.values

    model = LinearRegression()
    model.fit(X,y)

    future = model.predict([[len(job_counts)]])

    st.write("Predicted next demand level:", int(future[0]))