import numpy as np
import pandas as pd
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from typing import Optional, List

from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender


class HybridRecommender(AbstractSongRecommender):
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame,
                 k: int = 15, attribute_list: Optional[List[str]] = None,
                 content_weight: float = 0.7, collaborative_weight: float = 0.3,
                 vector_size: int = 100, window: int = 5, epochs: int = 5, sg: int = 0):
        """
        :param df_playlist: DataFrame containing playlist information.
        :param df_tracks: DataFrame containing detailed track information.
        :param k: Number of latent factors for collaborative filtering.
        :param attribute_list: List of song features for Word2Vec.
        :param content_weight: Weight for the content-based recommendation.
        :param collaborative_weight: Weight for the collaborative filtering recommendation.
        :param vector_size: Size of the word embedding vectors for Word2Vec.
        :param window: Maximum distance between the current and predicted word within a sentence for Word2Vec.
        :param epochs: Number of training epochs for Word2Vec.
        :param sg: Training algorithm for Word2Vec.
        """
        super().__init__(df_playlist, df_tracks)

        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        self.content_recommender = ContentRecommender(df_playlist, df_tracks,
                                                      attribute_list, vector_size, window,
                                                      epochs, sg)

        self.collaborative_recommender = CollaborativeRecommender(df_playlist, df_tracks, k)

    def recommend_tracks(self, playlist_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommends top N tracks for the given playlist based on both collaborative and content-based filtering.

        :param playlist_id: ID of the playlist for which recommendations are made.
        :param top_k: Number of recommended tracks.
        :return: DataFrame containing the top K recommended tracks, with collaborative rating and combined rating.
        """
        content_recs = self.content_recommender.recommend_tracks(playlist_id, top_k)
        collaborative_recs = self.collaborative_recommender.recommend_tracks(playlist_id, top_k)
        merged_recs = content_recs.copy()

        # Add the collaborative predicted ratings and the combined rating
        merged_recs['predicted_rating_collab'] = collaborative_recs['predicted_rating'].values
        merged_recs['combined_predicted_rating'] = (self.content_weight * merged_recs['predicted_rating'] +
                                                    self.collaborative_weight * merged_recs['predicted_rating_collab'])

        top_track_indices = np.argsort(merged_recs['combined_predicted_rating'])[::-1][:top_k]

        top_tracks = self.df_tracks.loc[top_track_indices]

        # Assign the predicted ratings to the top tracks
        top_tracks['predicted_rating_content'] = merged_recs['predicted_rating'].iloc[top_track_indices].values
        top_tracks['predicted_rating_collab'] = merged_recs['predicted_rating_collab'].iloc[top_track_indices].values
        top_tracks['predicted_rating'] = merged_recs['combined_predicted_rating'].iloc[
            top_track_indices].values

        return top_tracks
