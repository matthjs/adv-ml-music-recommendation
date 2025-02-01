from typing import Tuple, List, Dict, Any
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score
from joblib import Parallel, delayed
from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender
from adv_ml_music_recommendation.recommenders.hybridrecommender import HybridRecommender
from adv_ml_music_recommendation.util.data_functions import get_interacted_tracks, get_tracks_by_playlist_associate, \
    get_number_of_playlists
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender


class RecommenderEvaluator:
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, type: str = 'hybrid',
                 test_ratio=0.1):
        """
        Initializes the evaluator with playlists, tracks, and recommender type.
        """
        self.cached_content_rec = None

        self.df_playlist = df_playlist
        self.df_tracks = df_tracks
        self.type = type
        self.df_train, self.df_test = self._prepare_split_data(test_ratio)
        self.model = self._initialize_model(self.df_train, self.df_tracks)

    def evaluate(self):
        return self._evaluate_model(self.model, self.df_test)

    def perform_k_fold_cv_parallel(self, hyperparameter_sets, n_splits):
        playlist_groups = self.df_train.groupby('playlist_id')
        with Parallel(n_jobs=-1, backend="loky") as parallel:
            results = parallel(
                delayed(self._process_hyperparameter_set)(hyperparams, n_splits, playlist_groups)
                for hyperparams in hyperparameter_sets
            )

        return results

    def perform_k_fold_cv(
            self,
            hyperparameter_sets: List[Dict],
            n_splits: int
    ) -> List[Dict]:
        """
        Perform K-fold cross-validation with per-playlist track splitting.
        """
        results = []
        # Group tracks by playlist
        playlist_groups = self.df_train.groupby('playlist_id')

        for hyperparams in hyperparameter_sets:
            results.append(self._process_hyperparameter_set(hyperparams, n_splits, playlist_groups))

        logger.info("Performing K-fold cross done.")
        return results

    def _process_hyperparameter_set(self, hyperparams, n_splits, playlist_groups):
        logger.info(f'Performing K-fold cross for {hyperparams}')
        fold_metrics = {'precision': [], 'recall': [], 'accuracy': []}

        # Create K folds at the track level within each playlist
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Generate indices for splits across all playlists
        fold_splits = []
        for pl_id, pl_df in playlist_groups:
            tracks = pl_df['track_uri'].values
            # if len(tracks) < 2:
            #     # Skip playlists with insufficient tracks for splitting
            #     fold_splits.append([(tracks, np.array([]))] * n_splits)
            #    continue

            # Generate splits for this playlist's tracks
            splits = list(kf.split(tracks))
            fold_splits.append([
                (tracks[train_idx], tracks[test_idx])
                for train_idx, test_idx in splits
            ])

        # For each fold, aggregate splits across all playlists
        for fold_idx in range(n_splits):
            logger.info(f'Computing val metrics for fold {fold_idx}')
            train_chunks = []
            val_chunks = []

            # Process each playlist's split for this fold
            for pl_idx, (pl_id, pl_df) in enumerate(playlist_groups):
                if len(fold_splits[pl_idx][fold_idx][1]) == 0:
                    # Handle playlists with no validation tracks
                    train_chunks.append(pl_df)
                    continue

                # Split tracks for this playlist
                train_tracks, val_tracks = fold_splits[pl_idx][fold_idx]

                # Create split DataFrames
                train_split = pl_df[pl_df['track_uri'].isin(train_tracks)]
                val_split = pl_df[pl_df['track_uri'].isin(val_tracks)]

                train_chunks.append(train_split)
                val_chunks.append(val_split)

            # Combine splits across all playlists
            fold_train = pd.concat(train_chunks)
            fold_val = pd.concat(val_chunks)

            if fold_val.empty:
                continue

            model = self._initialize_model(fold_train, self.df_tracks, **hyperparams)

            # Evaluate on validation set
            metrics = self._evaluate_model(model, fold_val)
            # fold_metrics['precision'].append(metrics['average_precision'])
            # fold_metrics['recall'].append(metrics['average_recall'])
            fold_metrics['accuracy'].append(metrics['avg_accuracy'])

        # Aggregate results across folds
        if fold_metrics['accuracy']:
            return {
                'hyperparams': hyperparams,
                # 'average_precision': np.mean(fold_metrics['precision']),
                # 'average_recall': np.mean(fold_metrics['recall']),
                'average_accuracy': np.mean(fold_metrics['accuracy'])
            }

        return None

    def _initialize_model(self, df_train: pd.DataFrame, df_tracks: pd.DataFrame, **params):
        """
        Initializes the recommender model based on the type and additional hyperparameters.
        """
        if self.type == 'hybrid':
            return HybridRecommender(df_playlist=df_train, df_tracks=df_tracks, **params)
        elif self.type == 'collaborative':
            return CollaborativeRecommender(df_playlist=df_train, df_tracks=df_tracks, **params)
        elif self.type == 'content':
            rec = ContentRecommender(df_playlist=df_train, df_tracks=df_tracks, **params,
                                     cached_content_recommender=self.cached_content_rec)
            if self.cached_content_rec is None:
                self.cached_content_rec = rec
            return rec
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
