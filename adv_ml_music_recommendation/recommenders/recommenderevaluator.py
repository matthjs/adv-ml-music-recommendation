from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender


def get_interacted_tracks(
        tracks: pd.DataFrame,
        playlist_id: int,
        drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO: Check correctness
    """
    interacted_track_ids = set(tracks[tracks['playlist_id'] == playlist_id]['track_uri'])
    tracks_interacted = tracks[tracks['track_uri'].isin(interacted_track_ids)]
    tracks_not_interacted = tracks[~tracks['track_uri'].isin(interacted_track_ids)]

    if drop_duplicates is True:
        tracks_interacted = tracks_interacted.drop_duplicates(subset='track_uri', keep="first").reset_index()
        tracks_not_interacted = tracks_not_interacted.drop_duplicates(subset='track_uri', keep="first").reset_index()

    return tracks_interacted, tracks_not_interacted


class RecommenderEvaluator:
    """
    TODO: Add more evaluation metrics if needed.
    """

    def __init__(self, playlist_df: pd.DataFrame):
        self.playlist_df = playlist_df

    def evaluate_recommender_for_playlist(self, model: AbstractSongRecommender,
                                          playlist_id, n: int = 100, seed: int = 42):
        # We identify users with playlist
        # an interacted track is a track that is in the playlist
        tracks_interacted, tracks_not_interacted = get_interacted_tracks(
            self.playlist_df, playlist_id
        )

        # Split interacted tracks in a `train` and `test` split
        train, test = train_test_split(tracks_interacted, test_size=0.2, random_state=seed)

        ranked_recommendations_df = model.recommend_tracks(playlist_id)

        # TODO: Understand how this metric is computed
        hits_at_5_count, hits_at_10_count = 0, 0
        for index, row in test.iterrows():
            non_interacted_sample = tracks_not_interacted.sample(n, random_state=seed)
            evaluation_ids = [row['track_uri']] + non_interacted_sample['track_uri'].tolist()
            evaluation_recommendations_df = ranked_recommendations_df[
                ranked_recommendations_df['track_uri'].isin(evaluation_ids)]
            # Verifying if the current interacted item is among the Top-N recommended items
            hits_at_5_count += 1 if row['track_uri'] in evaluation_recommendations_df['track_uri'][:5].tolist() else 0
            hits_at_10_count += 1 if row['track_uri'] in evaluation_recommendations_df['track_uri'][:10].tolist() else 0

        playlist_metrics = {'n': n,
                            'evaluation_count': len(test),
                            'hits@5': hits_at_5_count,
                            'hits@10': hits_at_10_count,
                            'recall@5': hits_at_5_count / len(test),
                            'recall@10': hits_at_10_count / len(test),
                            }

        return playlist_metrics

    def evaluate_model(self, model: AbstractSongRecommender, model_name: str, n=100, seed=42):
        playlists = []
        for playlist_id in self.playlist_df['playlist_id'].unique():
            playlist_metrics = self.evaluate_recommender_for_playlist(model, playlist_id, n=n, seed=seed)
            playlist_metrics['playlist_id'] = playlist_id
            playlists.append(playlist_metrics)

        detailed_playlists_metrics = pd.DataFrame(playlists).sort_values('evaluation_count', ascending=False)

        global_recall_at_5 = detailed_playlists_metrics['hits@5'].sum() / detailed_playlists_metrics[
            'evaluation_count'].sum()
        global_recall_at_10 = detailed_playlists_metrics['hits@10'].sum() / detailed_playlists_metrics[
            'evaluation_count'].sum()

        global_metrics = {'model_name': model_name,  # make model_name member variable?
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          }

        return global_metrics, detailed_playlists_metrics
