nimport pandas as pd

from final_pipeline import analyze_resume


jd_df = pd.read_csv(
    'job_descriptions.csv'
)

resume = """

Experienced AI Engineer skilled in Python,
Machine Learning, NLP and TensorFlow.

"""

result = analyze_resume(
    resume,
    jd_df
)

print(result)