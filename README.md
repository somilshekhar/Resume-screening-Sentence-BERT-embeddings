# AI-Powered Resume Screening & Skill Matcher

![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project is an **AI-powered resume screening tool** that matches resumes with a given job description. It uses **NLP and embeddings** to calculate similarity, identify matched and missing skills, and provide AI-generated feedback. Perfect for **HR automation and recruiter-friendly applications**.

---

## Features

* Upload multiple PDF resumes.
* Paste a job description for screening.
* Extract text from resumes using **PyPDF2**.
* Generate **Sentence-BERT embeddings** (`all-MiniLM-L6-v2`) for job description and resumes.
* Calculate **cosine similarity** to rank candidates.
* Identify **matched and missing skills** from a predefined skill set:

  * Programming Languages
  * Data Science
  * Soft Skills
  * Databases
  * Cloud Platforms
  * Tools & Frameworks
  * Operating Systems
  * Web Development
  * Other
* Visualize **similarity scores with Plotly bar charts**.
* Generate **AI feedback** for each resume.
* Download results as a CSV file.

---

## Tech Stack

* **Python 3.10+**
* **Streamlit** – for interactive UI
* **PyPDF2** – PDF text extraction
* **Sentence-Transformers** – embeddings
* **Scikit-learn** – cosine similarity calculation
* **Pandas & Plotly** – data handling & visualization

---

## Installation

1. Clone the repository:

   ```bash
 git clone https://github.com/somilshekhar/Resume-screening-Sentence-BERT-embeddings.git
cd Resume-screening-Sentence-BERT-embeddings
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   source venv/bin/activate      # Linux/Mac
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Paste the **job description** in the text area.
2. Upload **PDF resumes**.
3. View similarity scores and skill matching details.
4. Generate **AI feedback**.
5. Download results as a **CSV file** for further analysis.

---

## Notes

* Predefined skills can be customized in `app.py`.
* Handles errors in PDF extraction, embedding generation, and similarity calculation.
* AI feedback is based on similarity percentage and skill counts.

---

## License

This project is licensed under the **MIT License**.

---

## Contact

**Developer:** Somil Shekhar
**Email:** [shekharsomil1192005@gmail.com](mailto:shekharsomil1192005@gmail.com)
**GitHub:** [somilshekhar](https://github.com/somilshekhar)
