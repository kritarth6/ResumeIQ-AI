import joblib
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# LOAD FILES

model = joblib.load(
    'resume_classifier.pkl'
)

tfidf = joblib.load(
    'tfidf_vectorizer.pkl'
)

jd_embeddings = joblib.load(
    'jd_embeddings.pkl'
)

all_jds = joblib.load(
    'jd_texts.pkl'
)

role_mapping = joblib.load(
    'role_mapping.pkl'
)

advanced_skills = joblib.load(
    'skills_database.pkl'
)


# LOAD EMBEDDING MODEL

embed_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)


# CLEAN TEXT

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'http\\S+', ' ', text)

    text = re.sub(r'\\S+@\\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\\s+', ' ', text)

    return text


# SKILL EXTRACTION

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in advanced_skills:

        if skill in text:

            found_skills.append(skill)

    return list(set(found_skills))


# CLEAN SKILL TEXT

def clean_skill_text(skill_text):

    skill_text = str(skill_text).lower()

    skill_text = re.sub(
        r'[^a-zA-Z, ]',
        '',
        skill_text
    )

    return skill_text


# MAIN PIPELINE

def analyze_resume(
    sample_resume,
    jd_df
):

    cleaned = clean_text(
        sample_resume
    )

    vector = tfidf.transform(
        [cleaned]
    )

    predicted_role = model.predict(
        vector
    )[0]

    final_role = role_mapping.get(
        predicted_role,
        predicted_role
    )

    resume_embedding = embed_model.encode(
        [sample_resume]
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

        "Predicted Role":
        final_role,

        "ATS Score":
        float(ats_score),

        "JD Match":
        float(round(best_score, 2)),

        "Skills":
        resume_skills,

        "Missing Skills":
        missing_skills,

        "Best JD":
        best_job_description[:500]

    }