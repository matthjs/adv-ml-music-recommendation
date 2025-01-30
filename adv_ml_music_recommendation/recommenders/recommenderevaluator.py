from typing import Tuple, List, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score

from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender
from adv_ml_music_recommendation.recommenders.hybridrecommender import HybridRecommender
from adv_ml_music_recommendation.recommenders.popularityrecommender import PopularityRecommender
from adv_ml_music_recommendation.util.data_functions import get_interacted_tracks, get_tracks_by_playlist_associate, \
    get_number_of_playlists
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender


class RecommenderEvaluator:
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, type: str = 'hybrid',
                 test_ratio=0.1):
        """
        Initializes the evaluator with playlists, tracks, and recommender type.
        """
        self.df_playlist = df_playlist
        self.df_tracks = df_tracks
        self.type = type
        self.df_train, self.df_test = self._prepare_split_data(test_ratio)
        self.model = self._initialize_model(self.df_train, self.df_tracks)

    def evaluate(self):
        return self._evaluate_model(self.model, self.df_test)

    def perform_k_fold_cv(
            self,
            hyperparameter_sets: List[Dict],
            n_splits: int
    ) -> List[Dict]:
        """
        Perform K-fold cross-validation with the correct splitting logic.

        :param hyperparameter_sets: List of hyperparameter dictionaries to evaluate.
        :param n_splits: Number of folds for cross-validation.
        :return: A list of results for each hyperparameter set with precision and recall metrics.
        """
        results = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for hyperparams in hyperparameter_sets:
            precision_scores = []
            recall_scores = []

            for train_idx, val_idx in kf.split(self.df_train):
                # Create train and validation sets based on playlist splitting
                df_train_split = self.df_train.iloc[train_idx]
                df_val_split = self.df_train.iloc[val_idx]

                # Adjust playlists for validation
                train_playlists = []
                val_playlists = []

                for _, row in df_train_split.iterrows():
                    # Split each playlist's tracks into train and test
                    train_tracks, val_tracks = train_test_split(
                        row['track_uri'], test_size=0.2, random_state=42
                    )
                    train_playlists.append({'playlist_id': row['playlist_id'], 'track_uri': train_tracks})
                    val_playlists.append({'playlist_id': row['playlist_id'], 'track_uri': val_tracks})

                # Convert splits into DataFrames
                adjusted_train = pd.DataFrame(train_playlists)
                adjusted_val = pd.DataFrame(val_playlists)

                # Train the recommender model with adjusted training set
                model = self._initialize_model(adjusted_train, self.df_tracks, hyperparams)

                # Evaluate on validation set
                precision, recall = self._evaluate_model(model, adjusted_val)
                precision_scores.append(precision)
                recall_scores.append(recall)

            # Aggregate results for this hyperparameter set
            results.append({
                'hyperparams': hyperparams,
                'average_precision': sum(precision_scores) / len(precision_scores),
                'average_recall': sum(recall_scores) / len(recall_scores),
            })

        return results

    def _initialize_model(self, df_train: pd.DataFrame, df_tracks: pd.DataFrame, **params):
        """
        Initializes the recommender model based on the type and additional hyperparameters.
        """
        if self.type == 'hybrid':
            return HybridRecommender(df_playlist=df_train, df_tracks=df_tracks, **params)
        elif self.type == 'collaborative':
            return CollaborativeRecommender(df_playlist=df_train, df_tracks=df_tracks, **params)
        elif self.type == 'content':
            return ContentRecommender(df_playlist=df_train, df_tracks=df_tracks, **params)
        else:
            raise ValueError(
                f"Invalid type: {self.type}. Must be one of 'hybrid', 'collaborative', or 'content'.")

    # def _split_playlist(self, playlist_tracks: List[str], test_ratio: float = 0.2):
    #     """
    #    Splits the tracks in a playlist into train and test sets.
    #    """
    #    split_index = int(len(playlist_tracks) * (1 - test_ratio))
    #    return playlist_tracks[:split_index], playlist_tracks[split_index:]

    def _prepare_split_data(self, test_ratio: float = 0.2):
        """
        Splits the playlists into train and test sets by removing tracks for testing.
        """
        train_data = []
        test_data = []

        for playlist_id, group in self.df_playlist.groupby('playlist_id'):
            track_uris = group['track_uri'].to_list()
            train_uris, test_uris = train_test_split(track_uris, test_size=test_ratio, random_state=42)

            train_data.extend({'track_uri': uri, 'playlist_id': playlist_id} for uri in train_uris)
            test_data.extend({'track_uri': uri, 'playlist_id': playlist_id} for uri in test_uris)

        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)

        return df_train, df_test

    def _evaluate_recommender_for_playlist(self, model, playlist_id, df_test: pd.DataFrame):
        """
        Evaluates the recommender system for a single playlist.
        """
        # Get the ground truth: test track URIs for the playlist
        test_track_uris = get_tracks_by_playlist_associate(df_test, playlist_id)

        # top-K recommendations
        k = len(test_track_uris)

        # Get recommendations from the train model
        ranked_recommendations_df = model.recommend_tracks(playlist_id, k)
        # Extract the recommended track URIs
        recommended_track_uris = ranked_recommendations_df['track_uri'].tolist()

        hits = len(set(recommended_track_uris) & set(test_track_uris))
        accuracy = hits / k

        return accuracy

    def _evaluate_model(self, model, df_test):
        """
        Evaluates the recommender system on a single train-test split.
        """
        total_accuracy = 0
        num_playlists = get_number_of_playlists(df_test)

        # Iterate over all playlists in the test_data
        for playlist_id in df_test['playlist_id'].unique():
            accuracy = self._evaluate_recommender_for_playlist(model, playlist_id, df_test)
            total_accuracy += accuracy

        avg_accuracy = total_accuracy / num_playlists

        return {
            'avg_accuracy': avg_accuracy
        }
