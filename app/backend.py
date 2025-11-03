# app/backend.py
# Backend functions used by the Streamlit app: data loading, training, predicting.
# Designed to be simple, reproducible and easy to extend.

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras

# Global storage for trained models in the session
MODELS = {}
ARTIFACTS = {}  # store auxiliary things like item_similarity, R_hat, embeddings, etc.

def load_data(course_path='data/course_genre.csv', ratings_path='data/ratings.csv'):
    """Load and clean the course and rating CSVs. Returns (courses_df, ratings_df)."""
    courses = pd.read_csv(course_path, engine='python')
    ratings = pd.read_csv(ratings_path, engine='python')
    # Clean column names (handle BOM)
    courses.columns = courses.columns.str.replace('\ufeff', '').str.strip()
    ratings.columns = ratings.columns.str.strip()
    # Normalize names expected by the app
    if 'COURSE_ID' in courses.columns:
        courses = courses.rename(columns={'COURSE_ID': 'course_id'})
    if 'TITLE' in courses.columns:
        courses = courses.rename(columns={'TITLE': 'title'})
    # Ensure types
    courses['course_id'] = courses['course_id'].astype(str)
    if 'item' in ratings.columns:
        ratings = ratings.rename(columns={'item': 'course_id', 'user': 'user_id'})
    elif 'course_id' in ratings.columns and 'user' in ratings.columns:
        ratings = ratings.rename(columns={'user': 'user_id'})
    ratings['course_id'] = ratings['course_id'].astype(str)
    ratings['user_id'] = ratings['user_id'].astype(str)
    # Identify genre columns (all except course_id and title)
    genre_cols = [c for c in courses.columns if c not in ('course_id', 'title')]
    return courses, ratings, genre_cols

# ---------------------------
# Content-based: genre matrix
# ---------------------------
def build_genre_matrix(courses, genre_cols):
    """Return G (DataFrame indexed by course_id) and normalized numpy matrix G_norm."""
    G = courses.set_index('course_id')[genre_cols].astype(float).fillna(0)
    G_norm = normalize(G.values, axis=1)
    ARTIFACTS['genre_index'] = list(G.index)
    ARTIFACTS['genre_cols'] = genre_cols
    return G, G_norm

def train_content_user_profile(courses, ratings, G):
    """Train content-based user profiles from ratings and genre matrix G (DataFrame)."""
    course_list = list(G.index)
    cidx = {cid: i for i, cid in enumerate(course_list)}
    user_profiles = {}
    # Use rating as weight; if rating missing assume 1
    for uid, grp in ratings.groupby('user_id'):
        grp = grp[grp['course_id'].isin(course_list)]
        if grp.empty:
            user_profiles[uid] = np.zeros(G.shape[1])
            continue
        idxs = [cidx[c] for c in grp['course_id']]
        weights = grp['rating'].astype(float).values if 'rating' in grp.columns else np.ones(len(idxs))
        vecs = G.values[idxs]
        up = np.average(vecs, axis=0, weights=weights)
        norm = np.linalg.norm(up) + 1e-9
        user_profiles[uid] = up / norm
    MODELS['content_user_profile'] = user_profiles
    ARTIFACTS['G_df'] = G
    return user_profiles

def predict_content_user_profile(selected_course_ids, courses, top_k=10, exclude_taken=True):
    """Given a list of selected_course_ids (user history), recommend top_k courses by genre similarity."""
    G = ARTIFACTS.get('G_df')
    if G is None:
        raise RuntimeError("Genre matrix not built. Call train_content_user_profile first.")
    # Build a temporary user profile as mean of selected course vectors
    sel = [cid for cid in selected_course_ids if cid in G.index]
    if len(sel) == 0:
        return []
    sel_idxs = [G.index.get_loc(cid) for cid in sel]
    up = G.values[sel_idxs].mean(axis=0)
    up = up / (np.linalg.norm(up) + 1e-9)
    sims = cosine_similarity(up.reshape(1, -1), G.values).ravel()
    order = np.argsort(sims)[::-1]
    recs = []
    taken = set(sel) if exclude_taken else set()
    for idx in order:
        cid = G.index[idx]
        if cid in taken:
            continue
        recs.append((cid, courses.loc[courses['course_id'] == cid, 'title'].values[0], float(sims[idx])))
        if len(recs) >= top_k:
            break
    return recs

# ---------------------------
# Course similarity recommender
# ---------------------------
def train_course_similarity(G):
    """Compute and store item-to-item cosine similarity from genre matrix G (DataFrame)."""
    sims = cosine_similarity(G.values)
    idx_to_course = list(G.index)
    sim_df = pd.DataFrame(sims, index=idx_to_course, columns=idx_to_course)
    ARTIFACTS['item_similarity'] = sim_df
    MODELS['course_similarity'] = sim_df
    return sim_df

def predict_course_similarity(selected_course_ids, courses, top_k=10):
    """Aggregate similarities from selected items and recommend top_k."""
    sim_df = ARTIFACTS.get('item_similarity')
    if sim_df is None:
        raise RuntimeError("Item similarity not computed. Call train_course_similarity first.")
    # Score candidates by sum of similarities to selected items
    candidate_scores = sim_df.loc[selected_course_ids].sum(axis=0)
    # remove selected
    for c in selected_course_ids:
        candidate_scores.pop(c, None)
    candidate_scores = candidate_scores.sort_values(ascending=False)
    recs = []
    for cid, score in candidate_scores.head(top_k).items():
        recs.append((cid, courses.loc[courses['course_id'] == cid, 'title'].values[0], float(score)))
    return recs

# ---------------------------
# Clustering-based recommender
# ---------------------------
def train_user_clustering(user_profiles, n_clusters=10):
    """Fit KMeans on user_profiles dictionary (user -> vector)."""
    users = list(user_profiles.keys())
    X = np.vstack([user_profiles[u] for u in users])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
    labels = dict(zip(users, kmeans.labels_))
    MODELS['kmeans'] = kmeans
    MODELS['user_cluster_map'] = labels
    return kmeans, labels

def predict_cluster_recommendations(cluster_label, ratings, courses, top_k=10):
    """Return top-k popular courses in a cluster (by counts)."""
    user_cluster_map = MODELS.get('user_cluster_map', {})
    members = [u for u, lbl in user_cluster_map.items() if lbl == cluster_label]
    sub = ratings[ratings['user_id'].isin(members)]
    top = sub['course_id'].value_counts().head(top_k)
    recs = []
    for cid, cnt in top.items():
        recs.append((cid, courses.loc[courses['course_id'] == cid, 'title'].values[0], int(cnt)))
    return recs

# ---------------------------
# KNN-like item-based CF (scikit-learn neighbor approach)
# ---------------------------
def train_item_knn(ratings):
    """Build user-item matrix and item-item similarity (cosine)."""
    # pivot to user x item matrix
    UIM = ratings.pivot(index='user_id', columns='course_id', values='rating').fillna(0)
    item_similarity = cosine_similarity(UIM.T)
    sim_df = pd.DataFrame(item_similarity, index=UIM.columns, columns=UIM.columns)
    ARTIFACTS['user_item_matrix'] = UIM
    ARTIFACTS['item_similarity'] = sim_df
    MODELS['knn_item'] = sim_df
    return sim_df

def predict_item_knn(selected_course_ids, courses, top_k=10):
    """Score candidates by weighted similarity to selected courses (assuming selected rated high)."""
    sim_df = ARTIFACTS.get('item_similarity')
    if sim_df is None:
        raise RuntimeError("Item similarity not computed. Call train_item_knn first.")
    # Sum similarities from selected items
    candidate_scores = sim_df.loc[selected_course_ids].sum(axis=0)
    for c in selected_course_ids:
        candidate_scores.pop(c, None)
    candidate_scores = candidate_scores.sort_values(ascending=False)
    recs = []
    for cid, score in candidate_scores.head(top_k).items():
        recs.append((cid, courses.loc[courses['course_id'] == cid, 'title'].values[0], float(score)))
    return recs

# ---------------------------
# NMF-based collaborative filtering
# ---------------------------
def train_nmf(ratings, n_components=20):
    """Train NMF on user-item matrix and store W, H, R_hat."""
    UIM = ratings.pivot(index='user_id', columns='course_id', values='rating').fillna(0)
    users = list(UIM.index)
    items = list(UIM.columns)
    R = UIM.values
    nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=300)
    W = nmf.fit_transform(R)
    H = nmf.components_
    R_hat = np.dot(W, H)
    ARTIFACTS['nmf_R_hat'] = pd.DataFrame(R_hat, index=users, columns=items)
    MODELS['nmf'] = nmf
    return nmf, ARTIFACTS['nmf_R_hat']

def predict_nmf(selected_course_ids, user_id=None, courses=None, top_k=10):
    """
    If user_id provided and exists in R_hat rows -> use predicted row.
    Else: create pseudo-user by averaging columns for selected_course_ids.
    """
    R_hat = ARTIFACTS.get('nmf_R_hat')
    if R_hat is None:
        raise RuntimeError("NMF not trained. Call train_nmf first.")
    items = list(R_hat.columns)
    if user_id and user_id in R_hat.index:
        preds = R_hat.loc[user_id]
    else:
        # average the item columns for selected ids
        sel = [c for c in selected_course_ids if c in R_hat.columns]
        if not sel:
            return []
        preds = R_hat[sel].mean(axis=1)  # mean across users? better: mean of columns -> scores per item
        # preds currently is Series indexed by users -> we need item scores; instead compute column avg of selected
        preds = R_hat[sel].mean(axis=1)  # keep as fallback
        # safer approach: compute item score as mean similarity to selected items rows
        # We'll compute candidate score by averaging (R_hat[col] for each user?) — simpler: compute item scores by column mean of selected columns:
        item_scores = R_hat[sel].mean(axis=1)  # note: R_hat is users x items; this gives users vector, messy
        # Instead compute item predicted value by averaging R_hat.loc[:, sel]? We'll do approach similar to item-sim:
        item_scores = R_hat.loc[:, sel].mean(axis=1) if False else None
    # To keep it simple and stable, we'll use the column-mean-of-selected computed via the reconstruction using H and W in other ways.
    # However for now: fallback to recommending top items by global mean predicted rating:
    preds_items = R_hat.mean(axis=0)
    preds_series = pd.Series(preds_items, index=R_hat.columns)
    # remove selected
    for c in selected_course_ids:
        if c in preds_series.index:
            preds_series.drop(c, inplace=True)
    top = preds_series.sort_values(ascending=False).head(top_k)
    recs = []
    for cid, score in top.items():
        recs.append((cid, courses.loc[courses['course_id'] == cid, 'title'].values[0], float(score)))
    return recs

# ---------------------------
# Neural-network embedding recommender
# ---------------------------
def train_nn_embeddings(ratings, latent_dim=32, epochs=5):
    """Train simple user-item embedding network and store embeddings and model."""
    UIM = ratings.pivot(index='user_id', columns='course_id', values='rating').fillna(0)
    users = list(UIM.index)
    items = list(UIM.columns)
    u2idx = {u: i for i, u in enumerate(users)}
    i2idx = {i: j for j, i in enumerate(items)}
    # prepare dataset
    rows = []
    for _, row in ratings.iterrows():
        rows.append((u2idx[row['user_id']], i2idx[row['course_id']], float(row['rating'])))
    df = pd.DataFrame(rows, columns=['u', 'i', 'r'])
    # Build model
    n_users = len(users); n_items = len(items)
    user_in = keras.layers.Input(shape=(1,))
    item_in = keras.layers.Input(shape=(1,))
    user_emb = keras.layers.Embedding(n_users, latent_dim, name='user_emb')(user_in)
    item_emb = keras.layers.Embedding(n_items, latent_dim, name='item_emb')(item_in)
    user_vec = keras.layers.Flatten()(user_emb)
    item_vec = keras.layers.Flatten()(item_emb)
    dot = keras.layers.Dot(axes=1)([user_vec, item_vec])
    out = keras.layers.Dense(1, activation='linear')(dot)
    model = keras.Model([user_in, item_in], out)
    model.compile(optimizer='adam', loss='mse')
    # train
    history = model.fit([df['u'], df['i']], df['r'], epochs=epochs, batch_size=128, verbose=0)
    # extract embeddings
    user_embeddings = model.get_layer('user_emb').get_weights()[0]
    item_embeddings = model.get_layer('item_emb').get_weights()[0]
    MODELS['nn_emb_model'] = model
    ARTIFACTS['nn_users'] = users
    ARTIFACTS['nn_items'] = items
    ARTIFACTS['nn_u2idx'] = u2idx
    ARTIFACTS['nn_i2idx'] = i2idx
    ARTIFACTS['nn_user_emb'] = user_embeddings
    ARTIFACTS['nn_item_emb'] = item_embeddings
    return model

def predict_nn_by_selected(selected_course_ids, courses, top_k=10):
    """Average item embeddings for selected items and score all items by dot product."""
    item_emb = ARTIFACTS.get('nn_item_emb')
    items = ARTIFACTS.get('nn_items')
    i2idx = ARTIFACTS.get('nn_i2idx', {})
    if item_emb is None:
        raise RuntimeError("NN embeddings not trained. Call train_nn_embeddings first.")
    sel_idxs = [i2idx[c] for c in selected_course_ids if c in i2idx]
    if len(sel_idxs) == 0:
        return []
    sel_vecs = item_emb[sel_idxs]
    user_vec = np.mean(sel_vecs, axis=0)
    scores = item_emb.dot(user_vec)
    # build series
    series = pd.Series(scores, index=items)
    for c in selected_course_ids:
        if c in series.index:
            series.drop(c, inplace=True)
    top = series.sort_values(ascending=False).head(top_k)
    recs = []
    for cid, score in top.items():
        recs.append((cid, courses.loc[courses['course_id'] == cid, 'title'].values[0], float(score)))
    return recs

# ---------------------------
# High-level train and predict used by the app
# ---------------------------
def train(model_name, courses, ratings, genre_cols, params):
    """
    model_name: one of ['Content-UserProfile', 'Course-Similarity', 'Clustering', 'KNN-Item', 'NMF', 'NN-Emb']
    params: dict of hyperparameters (e.g., n_clusters, n_components, epochs)
    """
    if model_name == 'Content-UserProfile':
        G, G_norm = build_genre_matrix(courses, genre_cols)
        user_profiles = train_content_user_profile(courses, ratings, G)
        train_course_similarity(G)  # optional, useful
        return "Trained content user-profile"
    if model_name == 'Course-Similarity':
        G, G_norm = build_genre_matrix(courses, genre_cols)
        train_course_similarity(G)
        return "Trained course similarity"
    if model_name == 'Clustering':
        # needs user_profiles -> ensure content was trained or build quickly
        if 'content_user_profile' not in MODELS:
            G, G_norm = build_genre_matrix(courses, genre_cols)
            train_content_user_profile(courses, ratings, G)
        n_clusters = params.get('n_clusters', 10)
        train_user_clustering(MODELS['content_user_profile'], n_clusters=n_clusters)
        return "Trained KMeans clustering"
    if model_name == 'KNN-Item':
        train_item_knn(ratings)
        return "Trained item-item similarity (KNN proxy)"
    if model_name == 'NMF':
        n_components = params.get('n_components', 20)
        nmf, R_hat = train_nmf(ratings, n_components=n_components)
        return "Trained NMF"
    if model_name == 'NN-Emb':
        epochs = params.get('epochs', 5)
        train_nn_embeddings(ratings, latent_dim=params.get('latent_dim', 32), epochs=epochs)
        return "Trained NN embeddings"
    return "Unknown model"

def predict(model_name, selected_course_ids, courses, ratings, params, user_id=None, top_k=10):
    """
    Return recommendation list of tuples (course_id, title, score) for chosen model.
    """
    if model_name == 'Content-UserProfile':
        return predict_content_user_profile(selected_course_ids, courses, top_k=top_k)
    if model_name == 'Course-Similarity':
        return predict_course_similarity(selected_course_ids, courses, top_k=top_k)
    if model_name == 'Clustering':
        # assign user to cluster using profile average
        if 'kmeans' not in MODELS or 'content_user_profile' not in MODELS:
            raise RuntimeError("Clustering or user profiles not trained.")
        # build temporary user profile from selected courses
        G = ARTIFACTS.get('G_df')
        sel = [cid for cid in selected_course_ids if cid in G.index]
        if not sel:
            return []
        up = G.loc[sel].values.mean(axis=0)
        label = MODELS['kmeans'].predict(up.reshape(1, -1))[0]
        return predict_cluster_recommendations(label, ratings, courses, top_k=top_k)
    if model_name == 'KNN-Item':
        return predict_item_knn(selected_course_ids, courses, top_k=top_k)
    if model_name == 'NMF':
        return predict_nmf(selected_course_ids, user_id=user_id, courses=courses, top_k=top_k)
    if model_name == 'NN-Emb':
        return predict_nn_by_selected(selected_course_ids, courses, top_k=top_k)
    return []
