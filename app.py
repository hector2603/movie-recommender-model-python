import json
from flask import Flask, request
import pandas as pd
from datetime import datetime
from RecommenderModel import RecommenderNet
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib

PROVIDERS_CSV = 'datasets/providers.csv'
MODEL_COLLABORATIVE_FILTERING = 'datasets/collaborativeFilteringModelVersion.csv'
DATASETS_NEW_MOVIES_CSV = 'datasets/newMovies.csv'
RATINGS_CSV = 'datasets/newRatings.csv'
FULL_RATINGS_CSV = 'datasets/ratings.csv'
PATH_MODELS = 'models/'

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Service up'


@app.route('/contentBased/getRecommendationsForUserId/<id>', methods=['GET'])
def contentBased_getRecommendationsForUSerId(id):

    df_newMovies = pd.read_csv(DATASETS_NEW_MOVIES_CSV, index_col="movieId")
    df_providers = pd.read_csv(PROVIDERS_CSV)
    df_providers.provider_id = df_providers.provider_id.apply(str)
    df_model_movies = df_newMovies["year"].copy()
    df_genres = df_newMovies['genres'].str.get_dummies(sep='|')
    df_genres = df_genres[
        ['Acción', 'Animación', 'Aventura', 'Bélica', 'Ciencia ficción', 'Comedia', 'Crimen', 'Documental', 'Drama',
         'Familia', 'Fantasía', 'Historia', 'Misterio', 'Música', 'Película de TV', 'Romance', 'Suspense', 'Terror',
         'Western']]
    df_model_movies = pd.merge(how='left', left=df_model_movies, right=df_genres, left_on='movieId', right_on='movieId')
    df_model_provider = df_providers.groupby(["movieId"]).agg({'provider_id': "|".join})
    df_model_provider = df_model_provider['provider_id'].str.get_dummies(sep='|')
    df_model_movies = pd.merge( how='left',left=df_model_movies, right=df_model_provider, left_on='movieId', right_on='movieId')

    new_rating_file_pd = pd.read_csv(RATINGS_CSV)
    new_rating_file_pd = new_rating_file_pd.astype({"userId": int}, errors='raise')
    new_rating_file_pd = new_rating_file_pd.astype({"movieId": int}, errors='raise')
    new_rating_file_pd = new_rating_file_pd.astype({"timestamp": int}, errors='raise')
    movies_watched_by_user = new_rating_file_pd[new_rating_file_pd.userId == int(id)]
    movies_not_watched = df_model_movies[~df_model_movies.index.isin(movies_watched_by_user.movieId.values)]
    movies_not_watched = movies_not_watched.fillna(0)
    X = movies_not_watched[['Acción', 'Animación', 'Aventura', 'Bélica', 'Ciencia ficción',
                         'Comedia', 'Crimen', 'Documental', 'Drama', 'Familia', 'Fantasía',
                         'Historia', 'Misterio', 'Música', 'Película de TV', 'Romance',
                         'Suspense', 'Terror', 'Western', '11', '119', '167', '190', '2', '3',
                         '31', '315', '337', '339', '350', '384', '444', '445', '467', '475',
                         '521', '531', '534', '546', '551', '554', '567', '569', '575', '619',
                         '67', '8']].values
    modelo_cargado = joblib.load(PATH_MODELS+'modelo_SVM_movies.pkl')  # Carga del modelo.
    y_pred_svm = modelo_cargado.predict(X)
    movies_not_watched['rating'] = y_pred_svm
    recommendation_dict = {"recommendedMovies": movies_not_watched.nlargest(10, 'rating').index.values.tolist()}
    return recommendation_dict


@app.route('/collaborativeFiltering/getRecommendationsForUserId/<id>', methods=['GET'])
def collaborativeFiltering_getRecommendationsForUSerId(id):
    new_rating_file_pd = pd.read_csv(RATINGS_CSV)
    new_rating_file_pd = new_rating_file_pd.astype({"userId": int}, errors='raise')
    new_rating_file_pd = new_rating_file_pd.astype({"movieId": int}, errors='raise')
    new_rating_file_pd = new_rating_file_pd.astype({"timestamp": int}, errors='raise')

    new_movies_file_pd = pd.read_csv(DATASETS_NEW_MOVIES_CSV)
    movie_ids = new_movies_file_pd["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    version_model_collaborative_filtering_pd = pd.read_csv(MODEL_COLLABORATIVE_FILTERING)
    last_version_name = version_model_collaborative_filtering_pd['file_name'][
        version_model_collaborative_filtering_pd.index[-1]]
    num_users = version_model_collaborative_filtering_pd['num_users'][
        version_model_collaborative_filtering_pd.index[-1]]
    num_movies = version_model_collaborative_filtering_pd['num_movies'][
        version_model_collaborative_filtering_pd.index[-1]]

    model = RecommenderNet(num_users, num_movies, 50)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.00005))
    model.build((17500067, 2))
    model.load_weights(PATH_MODELS + last_version_name, by_name=False, skip_mismatch=False, options=None)

    movies_watched_by_user = new_rating_file_pd[new_rating_file_pd.userId == int(id)]
    movies_not_watched = new_movies_file_pd[~new_movies_file_pd["movieId"].isin(movies_watched_by_user.movieId.values)][
        "movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_movie_array = np.hstack(([[int(id) - 1]] * len(movies_not_watched), movies_not_watched))

    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    recommendation_dict = {"recommendedMovies": recommended_movie_ids}
    return recommendation_dict


@app.route('/collaborativeFiltering/trainModel/', methods=['GET'])
def collaborativeFiltering_trainModel():
    new_rating_file_pd = pd.read_csv(RATINGS_CSV)
    new_rating_file_pd = new_rating_file_pd.astype({"userId": int}, errors='raise')
    new_rating_file_pd = new_rating_file_pd.astype({"movieId": int}, errors='raise')
    new_rating_file_pd = new_rating_file_pd.astype({"timestamp": int}, errors='raise')

    full_rating_file_pd = pd.read_csv(FULL_RATINGS_CSV)
    full_rating_file_pd = full_rating_file_pd.astype({"userId": int}, errors='raise')
    full_rating_file_pd = full_rating_file_pd.astype({"movieId": int}, errors='raise')
    full_rating_file_pd = full_rating_file_pd.astype({"timestamp": int}, errors='raise')

    full_rating_file_pd = pd.concat([new_rating_file_pd, full_rating_file_pd])

    user_ids = full_rating_file_pd["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    new_movies_file_pd = pd.read_csv(DATASETS_NEW_MOVIES_CSV)
    movie_ids = new_movies_file_pd["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    full_rating_file_pd["user"] = full_rating_file_pd["userId"].map(user2user_encoded)
    full_rating_file_pd["movie"] = full_rating_file_pd["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)

    full_rating_file_pd["rating"] = full_rating_file_pd["rating"].values.astype(np.float32)
    min_rating = min(full_rating_file_pd["rating"])
    max_rating = max(full_rating_file_pd["rating"])

    df = full_rating_file_pd.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    train_indices = int(0.7 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )

    model = RecommenderNet(num_users, num_movies, 50)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.00005)
    )
    model.build(x_train.shape)
    history = model.fit(x=x_train, y=y_train, batch_size=1024, epochs=1, verbose=2, validation_data=(x_val, y_val))

    version_model_collaborative_filtering_pd = pd.read_csv(MODEL_COLLABORATIVE_FILTERING)
    maxVersion = max(version_model_collaborative_filtering_pd["version"])

    new_model_name =  "modeloFiltadroColaborativo_{}".format(maxVersion)

    model_dict = {"version": (maxVersion + 1), "file_name": new_model_name,
                  "num_users": num_users, "num_movies": num_movies}
    version_model_collaborative_filtering_pd = version_model_collaborative_filtering_pd.append(model_dict,
                                                                                               ignore_index=True)
    version_model_collaborative_filtering_pd = version_model_collaborative_filtering_pd.astype({"version": int},
                                                                                               errors='raise')
    version_model_collaborative_filtering_pd = version_model_collaborative_filtering_pd.astype({"num_users": int},
                                                                                               errors='raise')
    version_model_collaborative_filtering_pd = version_model_collaborative_filtering_pd.astype({"num_movies": int},
                                                                                               errors='raise')
    version_model_collaborative_filtering_pd.to_csv(MODEL_COLLABORATIVE_FILTERING, index=False)

    model.save_weights(PATH_MODELS + new_model_name, overwrite=True, save_format=None, options=None)

    return "Finish Train moldel: "+new_model_name


@app.route('/addRaiting', methods=['POST'])
def add_raiting():
    json_data = request.get_json(force=True)
    _userId = json_data['userId']
    _movieId = json_data['movieId']
    _rating = json_data['rating']

    newRatingFile_pd = pd.read_csv(RATINGS_CSV)

    if _userId is None or _userId == "":
        maxIdUser = max(newRatingFile_pd["userId"])
        _userId = maxIdUser + 1

    dt = datetime.now()
    ts = datetime.timestamp(dt)
    user_dict = {"userId": _userId, "movieId": _movieId, "rating": _rating, "timestamp": ts}
    newRatingFile_pd_updated = newRatingFile_pd.append(user_dict, ignore_index=True)
    newRatingFile_pd_updated = newRatingFile_pd_updated.astype({"userId": int}, errors='raise')
    newRatingFile_pd_updated = newRatingFile_pd_updated.astype({"movieId": int}, errors='raise')
    newRatingFile_pd_updated = newRatingFile_pd_updated.astype({"timestamp": int}, errors='raise')
    newRatingFile_pd_updated.to_csv(RATINGS_CSV, index=False)
    return user_dict


@app.route('/addMovie', methods=['POST'])
def add_movie():
    json_data = request.get_json(force=True)
    _movieId = json_data['movieId']
    _title = json_data['title']
    _genres = json_data['genres']
    _year = json_data['year']

    newMovies_file_pd = pd.read_csv(DATASETS_NEW_MOVIES_CSV)

    movie_dict = {"movieId": _movieId, "title": _title, "genres": _genres, "year": _year}
    newMovies_file_pd_updated = newMovies_file_pd.append(movie_dict, ignore_index=True)
    newMovies_file_pd_updated = newMovies_file_pd_updated.astype({"movieId": int}, errors='raise')
    newMovies_file_pd_updated.to_csv(DATASETS_NEW_MOVIES_CSV, index=False)
    return movie_dict


@app.route('/addWatchProviderForMovie', methods=['POST'])
def add_watch_provider_for_movie():
    json_data = request.get_json(force=True)
    _movieId = json_data['movieId']
    _provider_id = json_data['provider_id']
    _provider_name = json_data['provider_name']
    _type = json_data['type']

    providers_file_pd = pd.read_csv(PROVIDERS_CSV)

    provider_dict = {"movieId": _movieId, "provider_id": _provider_id, "provider_name": _provider_name, "type": _type}
    providers_file_pd_update = providers_file_pd.append(provider_dict, ignore_index=True)
    providers_file_pd_update = providers_file_pd_update.astype({"movieId": int}, errors='raise')
    providers_file_pd_update = providers_file_pd_update.astype({"provider_id": int}, errors='raise')
    providers_file_pd_update.to_csv(PROVIDERS_CSV, index=False)
    return provider_dict


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
