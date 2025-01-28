from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender
from adv_ml_music_recommendation.recommenders.hybridrecommender import HybridRecommender
from adv_ml_music_recommendation.recommenders.popularityrecommender import PopularityRecommender
from adv_ml_music_recommendation.util.data_functions import get_interacted_tracks
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender

class RecommenderEvaluator:
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, type: str = 'hybrid'):
        self.train_data = []
        self.test_data = []

        # Group by playlist_id and perform train-test split for each playlist
        for playlist_id, group in df_playlist.groupby('playlist_id'):
            # Extract track_uris for the current playlist
            track_uris = group['track_uri'].tolist()

            # Perform train-test split
            train_uris, test_uris = train_test_split(track_uris, test_size=0.2, random_state=42)

            # Append the results to the train and test lists
            self.train_data.append({'playlist_id': playlist_id, 'track_uri': train_uris})
            self.test_data.append({'playlist_id': playlist_id, 'track_uri': test_uris})

        self.df_train = pd.DataFrame(self.train_data)
        self.df_test = pd.DataFrame(self.test_data)

        if type == 'hybrid':
            self.train_data_model = HybridRecommender(df_playlist=self.df_train, df_tracks=df_tracks)
            self.test_data_model = HybridRecommender(df_playlist=self.df_test, df_tracks=df_tracks)
        elif type == 'collaborative':
            self.train_data_model = CollaborativeRecommender(df_playlist=self.df_train, df_tracks=df_tracks)
            self.test_data_model = CollaborativeRecommender(df_playlist=self.df_test, df_tracks=df_tracks)
        elif type == 'content':
            self.train_data_model = ContentRecommender(df_playlist=self.df_train, df_tracks=df_tracks)
            self.test_data_model = ContentRecommender(df_playlist=self.df_test, df_tracks=df_tracks)
        elif type == 'popularity':
            self.train_data_model = PopularityRecommender(df_playlist=self.df_train, df_tracks=df_tracks)
            self.test_data_model = PopularityRecommender(df_playlist=self.df_test, df_tracks=df_tracks)
        else:
            raise ValueError(
                f"Invalid type: {type}. Must be one of 'hybrid', 'collaborative', 'content', 'popularity'.")


    def evaluate_recommender_for_playlist(self, playlist_id):
        # Get recommendations from the train model
        ranked_recommendations_df = self.train_data_model.recommend_tracks(playlist_id)

        # Extract the recommended track URIs
        recommended_track_uris = ranked_recommendations_df['track_uri'].tolist()

        # Get the ground truth: test track URIs for the playlist
        test_track_uris = self.df_test[self.df_test['playlist_id'] == playlist_id]['track_uri'].iloc[0]

        # Create binary vectors for precision and recall calculation
        y_true = [1 if uri in test_track_uris else 0 for uri in recommended_track_uris]
        y_pred = [1] * len(recommended_track_uris)  # All recommendations are predicted as relevant

        # Compute precision and recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        return precision, recall


    def evaluate_model(self):
        total_precision = 0
        total_recall = 0
        num_playlists = len(self.df_test)

        # Iterate over all playlists in the test_data
        for playlist_id in self.df_test['playlist_id'].unique():
            precision, recall = self.evaluate_recommender_for_playlist(playlist_id)
            total_precision += precision
            total_recall += recall

        # Compute average precision and recall
        avg_precision = total_precision / num_playlists
        avg_recall = total_recall / num_playlists

        return {
            'average_precision': avg_precision,
            'average_recall': avg_recall
        }


    def evaluate_recommender_for_playlist_old(self, playlist_id, n: int = 100, seed: int = 42):
        # an interacted track is a track that is in the playlist
        tracks_interacted, tracks_not_interacted = get_interacted_tracks(
            self.model.df_playlist, self.model.df_tracks, playlist_id
        )

        # Get recommendations based on a playlist
        ranked_recommendations_df = self.model.recommend_tracks(playlist_id)

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


    def evaluate_model_old(self, n=100, seed=42):
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
