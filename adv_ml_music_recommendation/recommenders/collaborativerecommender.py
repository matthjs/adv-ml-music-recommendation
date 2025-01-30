import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from adv_ml_music_recommendation.util.data_functions import create_sparse_interaction_matrix, \
    filter_out_playlist_tracks, get_tracks_by_playlist
from scipy.special import softmax


class CollaborativeRecommender(AbstractSongRecommender):
    """
    Matrix factorization based (using SVD) collaborative filtering.
    """

    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, k: int = 20):
        """
        :param df_playlist:
        :param df_tracks:
        :param k: number of latent factors (hyperparameter!)
        """
        super().__init__(df_playlist, df_tracks)
        # User-item interaction matrix R. Important the node that the sparse representation
        #         # allows us to store this insanely big (887060, 134712) matrix.
        self.playlist_tracks_matrix, self.playlist_id_mapping, self.track_id_mapping = create_sparse_interaction_matrix(
            df_playlist, df_tracks)

        # Construct predicted ratings matrix \hat{R}
        # Perform (partial) svd (for sparse matrices).
        self.u, self.s, self.vt = svds(self.playlist_tracks_matrix, k=k)

        # You could calculate the whole data matrix but for our dataset this takes up too much storage
        # (apparently 890GB for shape (887060, 134712)). Thankfully we can just compute the relevant rows
        # on the fly.

    def get_predicted_ratings_for_playlist(self, playlist_id: int, normalize: bool = True) -> np.ndarray:
        # Convert playlist_id to the correct row index
        if playlist_id not in self.playlist_id_mapping:
            raise ValueError(f"Playlist {playlist_id} not found in the interaction matrix.")
        row_idx = self.playlist_id_mapping[playlist_id]

        # Multiply the playlist row by the singular values
        playlist_ratings = self.u[row_idx, :] @ np.diag(self.s)

        # Multiply the result by vt to get the predicted ratings for all tracks for that playlist
        predicted_ratings = playlist_ratings @ self.vt

        if normalize:
            return softmax(predicted_ratings)

        return playlist_ratings

    def recommend_tracks(self, playlist_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommends top N tracks for the given playlist based on predicted ratings.
        """
        predicted_ratings = self.get_predicted_ratings_for_playlist(playlist_id)
        recommendations = pd.DataFrame({
            'id': list(self.track_id_mapping.keys()),  # Track IDs in correct order
            'predicted_rating': predicted_ratings
        }).merge(self.df_tracks, on='id', how='left')  # Merge with track details

        # Filter out tracks already in the playlist
        playlist_tracks = get_tracks_by_playlist(self.df_playlist, self.df_tracks, playlist_id)
        filtered_recommendations = filter_out_playlist_tracks(recommendations, playlist_tracks)

        # Sort by predicted rating and take the top K
        if top_k < 0:
            top_tracks = filtered_recommendations.sort_values(by='predicted_rating', ascending=False)
        else:
            top_tracks = filtered_recommendations.sort_values(by='predicted_rating', ascending=False).head(top_k)

        return top_tracks
