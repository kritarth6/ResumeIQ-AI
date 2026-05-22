import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="ResumeIQ AI",
    page_icon="🚀",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
        color: white;
    }

    .stApp {
        background: linear-gradient(to right, #0f172a, #111827);
    }

    .glass {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    .title {
        font-size: 50px;
        font-weight: bold;
        background: linear-gradient(to right, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-size: 20px;
        color: #cbd5e1;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# LOAD FILES
# -------------------------------------------------

model = joblib.load('resume_classifier.pkl')

tfidf = joblib.load('tfidf_vectorizer.pkl')
jd_df = pd.read_csv('small_job_descriptions.csv')

all_jds = jd_df[
    'Job Description'
].dropna().tolist()

role_mapping = joblib.load('role_mapping.pkl')

advanced_skills = joblib.load('skills_database.pkl')



embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'http\\S+', ' ', text)

    text = re.sub(r'\\S+@\\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\\s+', ' ', text)

    return text



def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in advanced_skills:

        if skill in text:

            found_skills.append(skill)

    return list(set(found_skills))



def clean_skill_text(skill_text):

    skill_text = str(skill_text).lower()

    skill_text = re.sub(r'[^a-zA-Z, ]', '', skill_text)

    return skill_text



def analyze_resume(sample_resume):

    cleaned = clean_text(sample_resume)

    vector = tfidf.transform([cleaned])

    predicted_role = model.predict(vector)[0]

    final_role = role_mapping.get(
        predicted_role,
        predicted_role
    )

    resume_embedding = embed_model.encode(
        [sample_resume]
    )

    jd_embeddings = embed_model.encode(
        all_jds
    )

    similarities = cosine_similarity(
        resume_embedding,
        jd_embeddings
    )

    best_match_index = similarities.argmax()

    best_score = similarities[
        0
    ][
        best_match_index
    ] * 100

    best_job_description = all_jds[
        best_match_index
    ]

    resume_skills = extract_skills(
        sample_resume
    )

    jd_skills_raw = jd_df[
        'skills'
    ].iloc[
        best_match_index
    ]

    cleaned_jd_skills = clean_skill_text(
        jd_skills_raw
    )

    jd_skill_list = [

        skill.strip()

        for skill in cleaned_jd_skills.split(',')

        if skill.strip()

    ]

    missing_skills = list(

        set(jd_skill_list)

        -

        set(resume_skills)

    )

    ats_score = min(

        round(
            best_score * 0.7 +
            len(resume_skills) * 2,
            2
        ),

        100

    )

    return {

        "Predicted Role": final_role,

        "ATS Score": float(ats_score),

        "JD Match": float(round(best_score, 2)),

        "Skills": resume_skills,

        "Missing Skills": missing_skills,

        "Best JD": best_job_description[:1000]

    }


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("🚀 ResumeIQ AI")

st.sidebar.markdown("AI Powered Resume Intelligence Platform")

st.sidebar.info(
    "Upload your resume and get ATS score, semantic job matching, skill analysis and AI insights."
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    "<div class='title'>ResumeIQ AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Resume Intelligence & ATS Optimization System</div>",
    unsafe_allow_html=True
)

st.markdown("---")


# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

resume_text = st.text_area(
    "Paste Resume Text",
    height=300,
    placeholder="Paste your resume here..."
)


analyze_btn = st.button("Analyze Resume")


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

if analyze_btn:

    if resume_text.strip() == "":

        st.warning("Please paste resume text")

    else:

        with st.spinner("Analyzing Resume using AI & NLP..."):

            result = analyze_resume(resume_text)

        st.success("Analysis Completed Successfully")


        # -----------------------------------------
        # METRICS
        # -----------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

            st.metric(
                "ATS Score",
                f"{result['ATS Score']}%"
            )

            st.markdown("</div>", unsafe_allow_html=True)


        with col2:

            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

            st.metric(
                "JD Match",
                f"{result['JD Match']}%"
            )

            st.markdown("</div>", unsafe_allow_html=True)


        with col3:

            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

            st.metric(
                "Predicted Role",
                result['Predicted Role']
            )

            st.markdown("</div>", unsafe_allow_html=True)


        st.markdown("---")


        # -----------------------------------------
        # CHARTS
        # -----------------------------------------

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['ATS Score'],
                title = {'text': "ATS Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "cyan"}
                }
            ))

            st.plotly_chart(fig, use_container_width=True)


        with chart_col2:

            fig2 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['JD Match'],
                title = {'text': "JD Match"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"}
                }
            ))

            st.plotly_chart(fig2, use_container_width=True)


        st.markdown("---")


        # -----------------------------------------
        # SKILLS
        # -----------------------------------------

        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:

            st.subheader("✅ Extracted Skills")

            for skill in result['Skills']:

                st.success(skill)


        with skill_col2:

            st.subheader("⚠ Missing Skills")

            if len(result['Missing Skills']) == 0:

                st.success("No major missing skills")

            else:

                for skill in result['Missing Skills']:

                    st.error(skill)


        st.markdown("---")


        # -----------------------------------------
        # BEST JD
        # -----------------------------------------

        st.subheader("📄 Best Matching Job Description")

        st.info(result['Best JD'])


        # -----------------------------------------
        # RADAR CHART
        # -----------------------------------------

        st.subheader("📊 Resume Analytics")

        categories = [
            'ATS',
            'JD Match',
            'Skills',
            'Experience',
            'Projects'
        ]

        values = [
            result['ATS Score'],
            result['JD Match'],
            len(result['Skills']) * 10,
            70,
            75
        ]

        radar_df = pd.DataFrame(dict(
            r=values,
            theta=categories
        ))

        fig3 = px.line_polar(
            radar_df,
            r='r',
            theta='theta',
            line_close=True
        )

        fig3.update_traces(fill='toself')

        st.plotly_chart(fig3, use_container_width=True)


        # -----------------------------------------
        # DOWNLOAD REPORT
        # -----------------------------------------

        st.download_button(
            label="Download Analysis Report",
            data=str(result),
            file_name="resume_analysis.txt",
            mime="text/plain"
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Built with AI, NLP, Machine Learning, Streamlit and Semantic Matching"
)
