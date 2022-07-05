import json
from flask import Flask, request
import pandas as pd
from datetime import datetime
from RecommenderModel import RecommenderNet
import numpy as np
from tensorflow.keras.models import Model


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Service up'


@app.route('/collaborativeFiltering/getRecommendationsForUserId/<id>', methods=['GET'])
def collaborativeFiltering_getRecommendationsForUSerId(id):
    model = RecommenderNet(num_users, num_movies, 50)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.00005)
    )

    model.load_weights("ModeloOctavaPrueba", by_name=False, skip_mismatch=False, options=None)

    return "fine"+id


@app.route('/addRaiting', methods=['POST'])
def add_raiting():
    json_data = request.get_json(force=True)
    _userId = json_data['userId']
    _movieId = json_data['movieId']
    _rating = json_data['rating']

    ratings_file = "datasets/newRatings.csv"
    newRatingFile_pd = pd.read_csv(ratings_file)

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
    newRatingFile_pd_updated.to_csv('datasets/newRatings.csv', index=False)
    return user_dict


@app.route('/addMovie', methods=['POST'])
def add_movie():
    json_data = request.get_json(force=True)
    _movieId = json_data['movieId']
    _title = json_data['title']
    _genres = json_data['genres']
    _year = json_data['year']

    newMovies_file = "datasets/newMovies.csv"
    newMovies_file_pd = pd.read_csv(newMovies_file)

    movie_dict = {"movieId": _movieId, "title": _title, "genres": _genres, "year": _year}
    newMovies_file_pd_updated = newMovies_file_pd.append(movie_dict, ignore_index=True)
    newMovies_file_pd_updated = newMovies_file_pd_updated.astype({"movieId": int}, errors='raise')
    newMovies_file_pd_updated.to_csv('datasets/newMovies.csv', index=False)
    return movie_dict


@app.route('/addWatchProviderForMovie', methods=['POST'])
def add_watch_provider_for_movie():
    json_data = request.get_json(force=True)
    _movieId = json_data['movieId']
    _provider_id = json_data['provider_id']
    _provider_name = json_data['provider_name']
    _type = json_data['type']

    providers_file = "datasets/providers.csv"
    providers_file_pd = pd.read_csv(providers_file)

    provider_dict = {"movieId": _movieId, "provider_id": _provider_id, "provider_name": _provider_name, "type": _type}
    providers_file_pd_update = providers_file_pd.append(provider_dict, ignore_index=True)
    providers_file_pd_update = providers_file_pd_update.astype({"movieId": int}, errors='raise')
    providers_file_pd_update = providers_file_pd_update.astype({"provider_id": int}, errors='raise')
    providers_file_pd_update.to_csv('datasets/providers.csv', index=False)
    return provider_dict


if __name__ == "__main__":
    app.run(host='0.0.0.0')
