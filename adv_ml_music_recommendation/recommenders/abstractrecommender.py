import pandas as pd
from abc import ABC, abstractmethod


class AbstractSongRecommender(ABC):
    """
    Contains core functionality of a recommender that
    recommends new songs to a playlist.
    """

    def __init__(self, tracks: pd.DataFrame):
        """
        Tracks == Entire dataset
        """
        self.tracks = tracks

    @abstractmethod
    def recommend_tracks(self, playlist_id: int, ignore_ids=None) -> pd.DataFrame:
        pass
