import pandas as pd
from abc import ABC, abstractmethod

from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender


class PopularityRecommender(AbstractSongRecommender):
    """
    The simplest recommender. Simple recommends the most popular songs.
    Based on the track_popularity metric.
    """
    def recommend_tracks(self, playlist_id: int, ignore_ids=None) -> pd.DataFrame:
        # Filters out tracks in the ignore_ids list, removes duplicates based on track_uri, resets the index,
        # and sorts the remaining tracks by popularity in descending order.
        if ignore_ids is None:
            ignore_ids = []
        recommendations_df = self.tracks[~self.tracks['track_uri'].isin(ignore_ids)] \
                                .drop_duplicates(subset='track_uri', keep="first").reset_index() \
                                .sort_values('track_popularity', ascending=False)

        return recommendations_df
