import pandas as pd
from abc import ABC, abstractmethod


class AbstractSongRecommender(ABC):
    """
    Contains core functionality of a recommender that
    recommends new songs to a playlist.
    """

    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame):
        """
        :param df_playlist: DataFrame containing 'track_uri' and 'playlist_id'.
        :param df_tracks: DataFrame containing detailed track information.
        """
        self.df_playlist = df_playlist
        self.df_tracks = df_tracks

    @abstractmethod
    def recommend_tracks(self, playlist_id: int, top_k: int = 10) -> pd.DataFrame:
        pass
