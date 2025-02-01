import pandas as pd
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from typing import Optional, List
from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender
from adv_ml_music_recommendation.util.data_functions import get_tracks_by_playlist, filter_out_playlist_tracks


class HybridRecommender(AbstractSongRecommender):
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame,
                 k: int = 20, attribute_list: Optional[List[str]] = None,
                 content_weight: float = 0.2,
                 vector_size: int = 50, window: int = 5, epochs: int = 10, sg: int = 0):
        """
        :param df_playlist: DataFrame containing playlist information.
        :param df_tracks: DataFrame containing detailed track information.
        :param k: Number of latent factors for collaborative filtering.
        :param attribute_list: List of song features for Word2Vec.
        :param content_weight: Weight for the content-based recommendation
        (collaborative weight is set to 1-content_weight)
        :param vector_size: Size of the word embedding vectors for Word2Vec.
        :param window: Maximum distance between the current and predicted word within a sentence for Word2Vec.
        :param epochs: Number of training epochs for Word2Vec.
        :param sg: Training algorithm for Word2Vec.
        """
        super().__init__(df_playlist, df_tracks)

        self.content_weight = content_weight
        self.collaborative_weight = 1 - content_weight

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
        # Get recommendations from both systems
        content_recs = self.content_recommender.recommend_tracks(playlist_id, -1)
        collab_recs = self.collaborative_recommender.recommend_tracks(playlist_id, -1)

        # Merge on track_uri with outer join
        merged = pd.merge(
            content_recs[['track_uri', 'predicted_rating']].rename(
                columns={'predicted_rating': 'content_score'}
            ),
            collab_recs[['track_uri', 'predicted_rating']].rename(
                columns={'predicted_rating': 'collab_score'}
            ),
            on='track_uri',
            how='outer'
        )
        # Fill missing scores with mean of existing values
        content_mean = merged['content_score'].mean() if not merged['content_score'].isnull().all() else 0
        collab_mean = merged['collab_score'].mean() if not merged['collab_score'].isnull().all() else 0
        merged['content_score'] = merged['content_score'].fillna(content_mean)
        merged['collab_score'] = merged['collab_score'].fillna(collab_mean)

        # Calculate combined score
        merged['combined_score'] = (
            self.content_weight * merged['content_score'] +
            self.collaborative_weight * merged['collab_score']
        )

        # Filter out playlist tracks
        playlist_tracks = get_tracks_by_playlist(self.df_playlist, self.df_tracks, playlist_id)
        merged = filter_out_playlist_tracks(merged, playlist_tracks)

        # Get top-k tracks and merge with full track details
        top_tracks = (
            merged.sort_values('combined_score', ascending=False)
            .head(top_k)
            .merge(self.df_tracks, on='track_uri', how='left')
        )

        return top_tracks[['track_uri', 'name', 'artists', 'content_score', 'collab_score', 'combined_score']]