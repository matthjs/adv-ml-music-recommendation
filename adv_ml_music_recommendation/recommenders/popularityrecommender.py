import pandas as pd
from abc import ABC, abstractmethod

from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender


class PopularityRecommender(AbstractSongRecommender):
    def recommend_tracks(self, playlist_id: str, ignore_ids=None) -> pd.DataFrame:
        pass
