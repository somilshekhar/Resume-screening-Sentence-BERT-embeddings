

import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


predefined_skills = {
    'Programming Languages': ['Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'Go', 'Swift', 'C#', 'PHP', 'R'],
    'Data Science': ['Machine Learning', 'Deep Learning', 'Statistical Analysis', 'Data Mining', 'Data Visualization', 'Big Data', 'Natural Language Processing', 'Computer Vision', 'Time Series Analysis', 'Experimental Design'],
    'Soft Skills': ['Communication', 'Teamwork', 'Problem Solving', 'Leadership', 'Adaptability', 'Critical Thinking', 'Collaboration', 'Presentation Skills', 'Negotiation', 'Emotional Intelligence'],
    'Databases': ['SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Cassandra', 'Oracle', 'SQLite'],
    'Cloud Platforms': ['AWS', 'Azure', 'Google Cloud Platform', 'Docker', 'Kubernetes', 'Serverless', 'Cloud Computing'],
    'Tools & Frameworks': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Spark', 'Hadoop', 'Git', 'Jupyter Notebooks', 'VS Code', 'Slack', 'JIRA', 'Agile', 'Scrum'],
    'Operating Systems': ['Linux', 'Windows', 'macOS'],
    'Web Development': ['HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'REST APIs'],
    'Other': ['Project Management', 'Business Intelligence', 'Technical Writing', 'Research', 'Consulting']
}


def extract_text_from_pdf(pdf_file):
  
    try:
        if isinstance(pdf_file, str):
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''  # Handle potential None return
        else:  # Assume file-like object
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or '' # Handle potential None return
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"


def generate_embeddings(text):
  
    # Initialize the SentenceTransformer model (use a pre-trained model)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the input text
    embedding = model.encode(text)

    return embedding


def calculate_similarity(embedding1, embedding2):
  
    # Reshape the embeddings to be 2D arrays as required by cosine_similarity
    embedding1_2d = embedding1.reshape(1, -1)
    embedding2_2d = embedding2.reshape(1, -1)

    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(embedding1_2d, embedding2_2d)

    # Extract the single similarity value from the 1x1 matrix
    similarity_score = similarity_matrix[0, 0]

    return similarity_score


def identify_skills(resume_text, predefined_skills):
  
    matched_skills = []
    all_predefined_skills = []

    # Create a flattened list of all predefined skills for easier checking
    for category, skills in predefined_skills.items():
        for skill in skills:
            all_predefined_skills.append(skill)

    # Convert resume text to lowercase for case-insensitive matching
    resume_text_lower = resume_text.lower()

    # Identify matched skills
    for skill in all_predefined_skills:
        if skill.lower() in resume_text_lower:
            matched_skills.append(skill)

    # Identify missing skills
    missing_skills = [skill for skill in all_predefined_skills if skill not in matched_skills]

    return matched_skills, missing_skills


st.set_page_config(page_title="AI-Powered Resume Screening & Skill Matcher", layout="wide")

st.title("AI-Powered Resume Screening & Skill Matcher")

job_description = st.text_area("Paste the Job Description here:")

uploaded_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)


if uploaded_files and job_description:
    st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")

    # Generate embedding for the job description
    job_embedding = generate_embeddings(job_description)

    results = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text.startswith("Error"):
            # Generate embedding for the resume text
            resume_embedding = generate_embeddings(resume_text)

            # Calculate similarity score
            similarity_score = calculate_similarity(job_embedding, resume_embedding)

            # Identify matched and missing skills
            matched_skills, missing_skills = identify_skills(resume_text, predefined_skills)

            results.append({
                "File Name": file_name,
                "Similarity Score": similarity_score,
                "Matched Skills": matched_skills,
                "Missing Skills": missing_skills
            })
        else:
            results.append({
                "File Name": file_name,
                "Error": resume_text
            })

    st.session_state['results'] = results

"""## Display results


Display the match percentages using a Plotly bar chart.
"""

if 'results' in st.session_state and st.session_state['results']:
    results_df = pd.DataFrame(st.session_state['results'])

    # Filter out entries with errors, handle case where 'Error' column might not exist or is None
    if 'Error' in results_df.columns:
        # Use .notna() on the 'Error' column to filter out rows where 'Error' is not NaN (i.e., where there is an error message)
        # Then invert the boolean mask to keep rows where 'Error' is NaN (i.e., no error)
        results_df_clean = results_df[results_df['Error'].isna()].copy()
    else:
        # If 'Error' column doesn't exist, assume no errors and use the original DataFrame
        results_df_clean = results_df.copy()


    if not results_df_clean.empty:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=results_df_clean['File Name'],
            y=results_df_clean['Similarity Score'] * 100,  # Convert to percentage
            marker_color='skyblue'
        ))

        fig.update_layout(
            title="Resume Similarity Scores",
            xaxis_title="Resume File",
            yaxis_title="Similarity Score (%)",
            yaxis=dict(range=[0, 100]) # Set y-axis range from 0 to 100
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid results to display. Please check for errors in file processing.")
else:
    st.info("Please upload PDF resumes and paste a job description to see the results.")

"""## Highlight skills

Show matched and missing skills for each resume.
"""

if 'results' in st.session_state and st.session_state['results']:
    st.subheader("Skill Matching Details")
    for result in st.session_state['results']:
        if 'Error' not in result:
            st.subheader(f"Skills for: {result['File Name']}")
            st.markdown("**Matched Skills:**")
            if result['Matched Skills']:
                for skill in result['Matched Skills']:
                    st.markdown(f"- {skill}")
            else:
                st.markdown("None")

            st.markdown("**Missing Skills:**")
            if result['Missing Skills']:
                for skill in result['Missing Skills']:
                    st.markdown(f"- {skill}")
            else:
                st.markdown("None")
        else:
            st.subheader(f"Processing Error for: {result['File Name']}")
            st.error(result['Error'])

"""## Generate ai feedback
 Generate AI feedback based on the match percentage and skills.
"""

if 'results' in st.session_state and st.session_state['results']:
    st.subheader("AI Feedback")
    if st.button("Generate AI Feedback"):
        for result in st.session_state['results']:
            st.subheader(f"Feedback for: {result['File Name']}")
            if 'Error' not in result:
                similarity = result['Similarity Score'] * 100
                matched_count = len(result['Matched Skills'])
                missing_count = len(result['Missing Skills'])

                feedback = f"Overall match: {similarity:.2f}%. "

                if similarity >= 75 and matched_count > missing_count:
                    feedback += "This resume is a strong match and contains many key skills. Consider this candidate highly."
                elif similarity >= 50 and matched_count > 0:
                    feedback += "This resume shows a good level of compatibility with the job description. Review the matched and missing skills for further assessment."
                elif similarity < 50 and matched_count == 0:
                    feedback += "This resume has a lower match score and lacks key skills. It may not be the best fit for this role."
                else:
                    feedback += "This resume has some relevant skills but the overall match is moderate. Further review is recommended."

                feedback += f" Matched skills: {matched_count}. Missing skills: {missing_count}."

                st.info(feedback)
            else:
                st.warning(f"Cannot generate feedback due to processing error: {result['Error']}")



# Add error handling for embedding generation and similarity calculation
if uploaded_files and job_description:
    st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")

    try:
        # Generate embedding for the job description
        job_embedding = generate_embeddings(job_description)
    except Exception as e:
        st.error(f"Error generating embedding for job description: {e}")
        job_embedding = None # Set to None if embedding fails

    results = []

    if job_embedding is not None: # Only proceed if job embedding was successful
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            try:
                resume_text = extract_text_from_pdf(uploaded_file)

                if not resume_text.startswith("Error"):
                    try:
                        # Generate embedding for the resume text
                        resume_embedding = generate_embeddings(resume_text)

                        try:
                            # Calculate similarity score
                            similarity_score = calculate_similarity(job_embedding, resume_embedding)

                            # Identify matched and missing skills
                            matched_skills, missing_skills = identify_skills(resume_text, predefined_skills)

                            results.append({
                                "File Name": file_name,
                                "Similarity Score": similarity_score,
                                "Matched Skills": matched_skills,
                                "Missing Skills": missing_skills
                            })
                        except Exception as e:
                            results.append({
                                "File Name": file_name,
                                "Error": f"Error calculating similarity or identifying skills: {e}"
                            })
                    except Exception as e:
                        results.append({
                            "File Name": file_name,
                            "Error": f"Error generating embedding for resume: {e}"
                        })
                else:
                    results.append({
                        "File Name": file_name,
                        "Error": resume_text
                    })
            except Exception as e:
                 results.append({
                    "File Name": file_name,
                    "Error": f"Error processing file: {e}"
                })


        st.session_state['results'] = results
    else:
        st.session_state['results'] = [] # Clear results if job embedding failed

# Display errors for individual files if any
if 'results' in st.session_state and st.session_state['results']:
    for result in st.session_state['results']:
        if 'Error' in result:
            st.error(f"Processing Error for {result['File Name']}: {result['Error']}")


# Handle empty job description
if not job_description:
    st.warning("Please enter a job description to proceed.")
    job_embedding = None
    st.session_state['results'] = [] # Clear results if job description is empty
else:
    try:
        # Generate embedding for the job description
        job_embedding = generate_embeddings(job_description)
    except Exception as e:
        st.error(f"Error generating embedding for job description: {e}")
        job_embedding = None # Set to None if embedding fails


if uploaded_files and job_embedding is not None:
    st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")
    results = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        try:
            resume_text = extract_text_from_pdf(uploaded_file)

            # Handle empty resume text
            if not resume_text or resume_text.startswith("Error"):
                results.append({
                    "File Name": file_name,
                    "Error": resume_text if resume_text.startswith("Error") else "No extractable text found in resume."
                })
            else:
                try:
                    # Generate embedding for the resume text
                    resume_embedding = generate_embeddings(resume_text)

                    try:
                        # Calculate similarity score
                        similarity_score = calculate_similarity(job_embedding, resume_embedding)

                        # Identify matched and missing skills
                        matched_skills, missing_skills = identify_skills(resume_text, predefined_skills)

                        results.append({
                            "File Name": file_name,
                            "Similarity Score": similarity_score,
                            "Matched Skills": matched_skills,
                            "Missing Skills": missing_skills
                        })
                    except Exception as e:
                        results.append({
                            "File Name": file_name,
                            "Error": f"Error calculating similarity or identifying skills: {e}"
                        })
                except Exception as e:
                    results.append({
                        "File Name": file_name,
                        "Error": f"Error generating embedding for resume: {e}"
                    })
        except Exception as e:
             results.append({
                "File Name": file_name,
                "Error": f"Error processing file: {e}"
            })

    st.session_state['results'] = results

elif uploaded_files and job_embedding is None:
    st.warning("Job description embedding failed. Cannot process resumes.")
    st.session_state['results'] = [] # Clear results if job embedding failed

else:
    st.session_state['results'] = [] # Clear results if no files uploaded or job description is empty

# Display errors for individual files if any
if 'results' in st.session_state and st.session_state['results']:
    for result in st.session_state['results']:
        if 'Error' in result:
            st.error(f"Processing Error for {result['File Name']}: {result['Error']}")



if 'results' in st.session_state and st.session_state['results']:
    # Create a DataFrame from the results
    results_df = pd.DataFrame(st.session_state['results'])

    # Filter out entries with errors before creating CSV, handle case where 'Error' column might not exist
    if 'Error' in results_df.columns:
        results_df_clean = results_df[results_df['Error'].isna()].copy()
    else:
        results_df_clean = results_df.copy()


    if not results_df_clean.empty:
        # Convert DataFrame to CSV string
        csv_data = results_df_clean.to_csv(index=False).encode('utf-8')

        # Provide a download button
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )

"""## Summary:

### Data Analysis Key Findings

*   The application successfully extracts text from uploaded PDF resumes and a job description using PyPDF2.
*   Sentence-BERT embeddings ('all-MiniLM-L6-v2' model) are generated for both the job description and each resume.
*   Cosine similarity is calculated between the job description embedding and each resume embedding to determine a match score.
*   Matched and missing skills are identified in each resume based on a predefined dictionary of skills.
*   Results, including file name, similarity score, matched skills, and missing skills, are stored in the Streamlit session state.
*   A Plotly bar chart visualizes the similarity scores (as percentages) for all successfully processed resumes.
*   Detailed lists of matched and missing skills are displayed for each resume.
*   Optional AI feedback is generated based on the similarity score and skill counts, providing a brief assessment of the resume's suitability.
*   Robust error handling is implemented for text extraction, embedding generation, similarity calculation, and skill identification, displaying specific error messages for problematic files.
*   Users can download the clean analysis results (excluding entries with errors) as a CSV file.

"""