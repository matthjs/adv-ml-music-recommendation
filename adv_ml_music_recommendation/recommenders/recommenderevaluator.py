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
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, type: str = 'collaborative'):
        self.df_train_data = []
        self.df_test_data = []
        self.df_tracks = df_tracks

        # Group by playlist_id and perform train-test split for each playlist
        for playlist_id, group in df_playlist.groupby('playlist_id'):
            '''
            if playlist_id > 1000:
                break
            '''
            if playlist_id % 10000 == 0:
                print(playlist_id)

            if group.shape[0] > 10:
                # Extract track_uris for the current playlist
                track_uris = group['track_uri'].tolist()

                # Perform train-test split
                train_uris, test_uris = train_test_split(track_uris, test_size=0.2, random_state=42)

                # Append the results to the train and test lists
                for uri in train_uris:
                     self.df_train_data.append({'playlist_id': playlist_id, 'track_uri': uri})

                for uri in test_uris:
                     self.df_test_data.append({'playlist_id': playlist_id, 'track_uri': uri})


        self.df_train_data = pd.DataFrame(self.df_train_data)
        self.df_test_data = pd.DataFrame(self.df_test_data)

        # Filter the tracks dataframe to keep only tracks that appear in the playlists dataframe:
        # & operator feels like questionable design
        print(self.df_train_data)
        #self.df_tracks = self.df_tracks[self.df_tracks['track_uri'].isin(self.df_train_data['track_uri']) & self.df_tracks['track_uri'].isin(self.df_test_data['track_uri'])]
        self.df_tracks = self.df_tracks[self.df_tracks['track_uri'].isin(self.df_train_data['track_uri'])]

        if type == 'hybrid':
            self.train_data_model = HybridRecommender(df_playlist=self.df_train_data, df_tracks=self.df_tracks)
            self.test_data_model = HybridRecommender(df_playlist=self.df_test_data, df_tracks=self.df_tracks)
        elif type == 'collaborative':
            self.train_data_model = CollaborativeRecommender(df_playlist=self.df_train_data, df_tracks=self.df_tracks)
            self.test_data_model = CollaborativeRecommender(df_playlist=self.df_test_data, df_tracks=self.df_tracks)
        elif type == 'content':
            self.train_data_model = ContentRecommender(df_playlist=self.df_train_data, df_tracks=self.df_tracks)
            self.test_data_model = ContentRecommender(df_playlist=self.df_test_data, df_tracks=self.df_tracks)
        elif type == 'popularity':
            self.train_data_model = PopularityRecommender(df_playlist=self.df_train_data, df_tracks=self.df_tracks)
            self.test_data_model = PopularityRecommender(df_playlist=self.df_test_data, df_tracks=self.df_tracks)
        else:
            raise ValueError(
                f"Invalid type: {type}. Must be one of 'hybrid', 'collaborative', 'content', 'popularity'.")


    def evaluate_recommender_for_playlist(self, playlist_id):
        # Get recommendations from the train model
        ranked_recommendations_df = self.train_data_model.recommend_tracks(playlist_id)

        # Extract the recommended track URIs
        recommended_track_uris = ranked_recommendations_df['track_uri'].tolist()

        # Get the ground truth: test track URIs for the playlist
        test_track_uris = self.df_test_data[self.df_test_data['playlist_id'] == playlist_id]['track_uri'].iloc[0]

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
        num_playlists = len(self.df_test_data)

        # Iterate over all playlists in the test_data
        for playlist_id in self.df_test_data['playlist_id'].unique():
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
