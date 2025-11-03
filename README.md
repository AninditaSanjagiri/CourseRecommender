# CourseRecommender
Suggest courses to users based on past ratings and genre similarities (hybrid of content-based and collaborative filtering).

🎓 Personalized Course Recommender System

A machine learning-based course recommendation app that suggests relevant online courses to users based on their interests and previous learning history.
This project combines content-based filtering, clustering, and collaborative filtering (KNN, NMF, neural embeddings) into one interactive Streamlit web app.

🧠 Project Overview

This project was developed as part of the IBM Machine Learning Capstone.
The goal was to design a personalized online course recommender system using multiple ML techniques:

Category	Technique	Description
Content-based	User Profile & Course Similarity	Recommends courses similar to what the user has already taken, using genre vectors.
Unsupervised	User Clustering (K-Means)	Groups users with similar preferences and recommends top courses from their cluster.
Collaborative Filtering	KNN, NMF, Neural Embeddings	Learns from historical user-item interactions to predict new preferences.

All models are trained and served through a Streamlit front-end that allows users to:

Select courses they’ve completed

Choose which model to use

Train and generate new recommendations instantly

🗂️ Project Structure
CourseRecommender-1/
│
├── app/
│   ├── backend.py              # Machine learning logic: training + prediction
│   ├── recommender_app.py      # Streamlit frontend
│
├── data/
│   ├── course_genre.csv        # Course catalog with genre columns
│   ├── ratings.csv             # User–course ratings
│
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation (this file)
└── venv/                       # Local virtual environment (not pushed to GitHub)

⚙️ Installation Guide
1. Clone or download the repository
git clone https://github.com/<your-username>/CourseRecommender-1.git
cd CourseRecommender-1

2. Create and activate a virtual environment

Windows

python -m venv venv
venv\Scripts\activate


macOS/Linux

python3 -m venv venv
source venv/bin/activate

3. Install required libraries
pip install -r requirements.txt


(If st-aggrid fails, install it manually as: pip install streamlit-aggrid)

▶️ Running the App

Once setup is done, start the Streamlit server:

streamlit run app/recommender_app.py


Then open the local URL displayed in the terminal (usually http://localhost:8501
).

🧩 How It Works

Select courses you’ve already taken from the data grid.

Pick a recommender model from the sidebar.

Adjust hyperparameters (number of clusters, top-K, epochs, etc.).

Click Train Model → then Recommend New Courses.

The app will display a table of recommended courses, along with a CSV download option.

🧮 Models Implemented
Model	Algorithm	Description
Content-UserProfile	Cosine similarity on genre vectors	Builds a profile from liked courses and recommends similar ones
Course-Similarity	Item–item cosine similarity	Finds courses most similar to a given course
Clustering	K-Means	Groups users and recommends top courses per cluster
KNN-Item	Nearest Neighbors	Learns item–item similarities from ratings
NMF	Non-Negative Matrix Factorization	Learns hidden latent factors to predict unseen ratings
NN-Emb	Neural Embedding Model (Keras)	Learns user/item vector embeddings to predict new preferences
📊 Example Workflow (in Streamlit)

Load dataset and explore courses

Select completed courses

Choose model — for example, Neural Embedding

Train → get recommendations

Download your personalized course list

💡 Key Features

Hybrid recommendation: content + collaborative approaches

Fully interactive Streamlit interface

Keras-based neural recommender with embeddings

Dynamic hyperparameter control from the sidebar

Ready for local or cloud deployment

Compatible with Coursera/IBM Capstone grading criteria

🧾 Requirements
Library	Version (recommended)
Python	3.9 – 3.11
streamlit	1.51.0
streamlit-aggrid	1.0.4.post3
scikit-learn	1.7.2
tensorflow	2.20.0
pandas	2.3.3
numpy	2.3.4
matplotlib	latest
wordcloud	latest
🏁 Future Improvements

Deploy via Streamlit Cloud or HuggingFace Spaces

Add user authentication and persistent profiles

Incorporate NLP-based similarity using course descriptions

Introduce hybrid scoring (weighted blend of models)

👩‍💻 Author

Ani
Machine Learning Enthusiast | Data Science Student

GitHub: github.com/AninditaSanjagiri

Email: anindita.sanjagiri@gmail.com