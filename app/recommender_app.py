# app/recommender_app.py
# Streamlit application for Course Recommender

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from backend import load_data, train, predict

st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("Personalized Course Recommender ")

# --- 1. Load data ---
@st.cache_data
def load_all(course_path='data/course_genre.csv', ratings_path='data/ratings.csv'):
    courses, ratings, genre_cols = load_data(course_path, ratings_path)
    return courses, ratings, genre_cols

courses, ratings, genre_cols = load_all()

# Left column: course picker (ag-grid)
st.sidebar.header("1. Select courses you've completed")
gb = GridOptionsBuilder.from_dataframe(courses[['course_id', 'title'] + genre_cols])
gb.configure_selection(selection_mode='multiple', use_checkbox=True)
gb.configure_pagination()
grid_options = gb.build()
grid_response = AgGrid(courses[['course_id', 'title'] + genre_cols], gridOptions=grid_options,
                       update_mode=GridUpdateMode.SELECTION_CHANGED, height=400, fit_columns_on_grid_load=True)
selected = grid_response['selected_rows']
selected_df = pd.DataFrame(selected)

st.sidebar.write(f"Selected courses: {selected_df.shape[0]}")

# --- 2. Model selection ---
backend_models = ['Content-UserProfile', 'Course-Similarity', 'Clustering', 'KNN-Item', 'NMF', 'NN-Emb']
model_selection = st.sidebar.selectbox("Choose recommender model", backend_models)

# --- 3. Hyper-parameters for each model ---
params = {}
st.sidebar.subheader("3. Hyper-parameters")

if model_selection == 'Content-UserProfile':
    params['top_k'] = st.sidebar.slider('Top-K', 1, 50, 10)
elif model_selection == 'Course-Similarity':
    params['top_k'] = st.sidebar.slider('Top-K', 1, 50, 10)
    params['sim_threshold'] = st.sidebar.slider('Similarity threshold (%)', 0, 100, 50, step=5)
elif model_selection == 'Clustering':
    params['n_clusters'] = st.sidebar.slider('Number of clusters', 2, 30, 8)
elif model_selection == 'KNN-Item':
    params['top_k'] = st.sidebar.slider('Top-K', 1, 50, 10)
elif model_selection == 'NMF':
    params['n_components'] = st.sidebar.slider('Latent dims (NMF)', 5, 100, 20)
elif model_selection == 'NN-Emb':
    params['latent_dim'] = st.sidebar.slider('Embedding size', 8, 128, 32)
    params['epochs'] = st.sidebar.slider('Training epochs', 1, 20, 5)

# --- 4. Training UI ---
st.sidebar.subheader("4. Training")
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.empty()
if training_button:
    training_text.text("Training... please wait")
    msg = train(model_selection, courses, ratings, genre_cols, params)
    training_text.text(f"{msg} (model: {model_selection})")

# --- 5. Prediction UI ---
st.sidebar.subheader("5. Prediction")
pred_button = st.sidebar.button("Recommend New Courses")

# Build selected_course_ids
selected_course_ids = selected_df['course_id'].tolist() if not selected_df.empty else []

if pred_button:
    if len(selected_course_ids) == 0:
        st.sidebar.warning("Please select at least one course you have taken.")
    else:
        recs = predict(model_selection, selected_course_ids, courses, ratings, params, user_id=None, top_k=params.get('top_k', 10))
        if not recs:
            st.sidebar.info("No recommendations found. Ensure you trained the model and selected courses exist in catalog.")
        else:
            rec_df = pd.DataFrame(recs, columns=['course_id', 'title', 'score'])
            st.subheader("Recommended Courses")
            st.dataframe(rec_df)
            csv = rec_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations (CSV)", csv, file_name='recommendations.csv', mime='text/csv')

# --- 6. Helpful status and artifacts ---
st.sidebar.markdown("---")
st.sidebar.markdown("Generated artifacts (from preprocessing)")
try:
    from pathlib import Path
    if Path("outputs/genre_counts.png").exists():
        st.sidebar.image("outputs/genre_counts.png", use_column_width=True)
except Exception:
    pass

st.markdown("### Dataset summary")
c1, c2 = st.columns(2)
with c1:
    st.write("Courses:", len(courses))
    st.write("Genres:", len(genre_cols))
with c2:
    st.write("Ratings rows:", len(ratings))
    st.write("Unique users:", ratings['user_id'].nunique())

st.info("Workflow: Select courses you completed → choose model → train model (if needed) → Recommend New Courses.")

