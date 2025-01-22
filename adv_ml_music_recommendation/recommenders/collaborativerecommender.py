import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from scipy.sparse.linalg import svds

from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from adv_ml_music_recommendation.util.data_functions import create_sparse_interaction_matrix


class CollaborativeRecommender(AbstractSongRecommender):
    """
    Matrix factorization based (using SVD) collaborative filtering.
    """

    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, k: int = 15):
        """
        :param df_playlist:
        :param df_tracks:
        :param k: number of latent factors (hyperparameter!)
        """
        super().__init__(df_playlist, df_tracks)
        # User-item interaction matrix R. Important the node that the sparse representation
        #         # allows us to store this insanely big (887060, 134712) matrix.
        self.playlist_tracks_matrix = create_sparse_interaction_matrix(df_playlist, df_tracks)

        # Construct predicted ratings matrix \hat{R}
        # Perform (partial) svd (for sparse matrices).
        self.u, self.s, self.vt = svds(self.playlist_tracks_matrix, k=k)

        # You could calculate the whole data matrix but for our dataset this takes up too much storage
        # (apparently 890GB for shape (887060, 134712)). Thankfully we can just compute the relevant rows
        # on the fly.

    def get_predicted_ratings_for_playlist(self, playlist_id: int, normalize: bool = True) -> np.ndarray:
        # Multiply the playlist row by the singular values
        playlist_ratings = self.u[playlist_id, :] @ np.diag(self.s)

        # Multiply the result by vt to get the predicted ratings for all tracks for that playlist
        predicted_ratings = playlist_ratings @ self.vt

        if normalize:
            min_rating = np.min(predicted_ratings)
            max_rating = np.max(predicted_ratings)
            normalized_ratings = (predicted_ratings - min_rating) / (max_rating - min_rating)
            return normalized_ratings

        return playlist_ratings

    def recommend_tracks(self, playlist_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommends top N tracks for the given playlist based on predicted ratings.
        """
        # Get predicted ratings for the playlist
        predicted_ratings = self.get_predicted_ratings_for_playlist(playlist_id)

        # Get the indices of the top K tracks with the highest predicted ratings
        top_track_indices = np.argsort(predicted_ratings)[::-1][:top_k]

        # Get the track IDs for the top K tracks
        top_tracks = self.df_tracks.loc[top_track_indices]

        # Get predicted ratings for the top K tracks
        top_tracks['predicted_rating'] = predicted_ratings[top_track_indices]

        # top_tracks[['track_uri', 'track_name', 'artist_name', 'predicted_rating']]
        return top_tracks
