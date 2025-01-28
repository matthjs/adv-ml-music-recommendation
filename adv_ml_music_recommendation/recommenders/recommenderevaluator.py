from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from adv_ml_music_recommendation.util.data_functions import get_interacted_tracks
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender



class RecommenderEvaluator:
    def __init__(self, model: AbstractSongRecommender, model_name: str):
        self.model = model
        self.model_name = model_name


    def evaluate_recommender_for_playlist(self, playlist_id, n: int = 100, seed: int = 42):
        # an interacted track is a track that is in the playlist
        tracks_interacted, tracks_not_interacted = get_interacted_tracks(
            self.model.df_playlist, self.model.df_tracks, playlist_id
        )

        # Split interacted tracks in a `train` and `test` split;
        train, test = train_test_split(tracks_interacted, test_size=0.2, random_state=seed)

        # train not used!
        #test = tracks_interacted

        # Get recommendations based on a playlist
        ranked_recommendations_df = self.model.recommend_tracks(playlist_id) # should be train

        playlist_metrics = []
        for top_N in range(2, 11):
            hits = 0

            # check for each track whether it appears in the model's top_N recommendations given
            # evaluation playlist
            for index, row in test.iterrows():
                # Sample tracks not present in the playlist
                non_interacted_sample = tracks_not_interacted.sample(n, random_state=seed)

                # Create evaluation set by combining current track with tracks unknown to user
                evaluation_ids = [row['track_uri']] + non_interacted_sample['track_uri'].tolist()

                # Get the intersection of ranked recommendations and evaluation ids set
                ## I do not understand how the track at 'row['track_uri']' can appear
                ## in the ranked recommendations_df since it a part of the original playlist
                evaluation_recommendations_df = ranked_recommendations_df[
                    ranked_recommendations_df['track_uri'].isin(evaluation_ids)]

                # Verifying if the track is among the Top-N count recommended items
                hits += 1 if row['track_uri'] in evaluation_recommendations_df['track_uri'][:top_N].tolist() else 0

            playlist_metrics.append({'top_N': top_N,
                                'evaluation_count': len(test),
                                'hits': hits,
                                'precision': hits / top_N,
                                'recall': hits / len(test)
                                })



        return playlist_metrics


    def evaluate_model(self, n=100, seed=42):
        playlists = []
        for playlist_id in self.model.df_playlist['playlist_id'].unique():
            playlist_metrics = self.evaluate_recommender_for_playlist(playlist_id, n=n, seed=seed)
            playlist_metrics['playlist_id'] = playlist_id
            playlists.append(playlist_metrics)

        detailed_playlists_metrics = pd.DataFrame(playlists).sort_values('evaluation_count', ascending=False)

        global_recall_at_5 = detailed_playlists_metrics['hits@5'].sum() / detailed_playlists_metrics[
            'evaluation_count'].sum()
        global_recall_at_10 = detailed_playlists_metrics['hits@10'].sum() / detailed_playlists_metrics[
            'evaluation_count'].sum()

        global_metrics = {'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          }

        return global_metrics, detailed_playlists_metrics
