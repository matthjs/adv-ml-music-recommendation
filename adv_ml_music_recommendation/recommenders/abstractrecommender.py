import pandas as pd
from abc import ABC, abstractmethod


class AbstractSongRecommender(ABC):
    """
    Contains core functionality of a recommender that
    recommends new songs to a playlist.
    """

    def __init__(self, playlist: pd.DataFrame):
        """
        TODO: check this
        """
        self.playlist = playlist
        self.tracks = None  # extract tracks from playlist dataframe?

    @abstractmethod
    def recommend_tracks(self, playlist_id: str, ignore_ids=None) -> pd.DataFrame:
        pass

