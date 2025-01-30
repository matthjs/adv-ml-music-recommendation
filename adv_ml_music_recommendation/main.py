import pandas as pd
from adv_ml_music_recommendation.recommenders.hybridrecommender import HybridRecommender
from adv_ml_music_recommendation.recommenders.recommenderevaluator import RecommenderEvaluator
from adv_ml_music_recommendation.util.data_functions import filter_playlist_df_min_tracks
from itertools import product


def main() -> None:
    df_playlist = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist, min_tracks=69)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    recommender = HybridRecommender(df_playlist=df_playlist, df_tracks=df_tracks)

    # Get the top 10 recommended tracks for a specific playlist
    playlist_id = 33660  # Replace with the playlist_id for which you want recommendations
    top_k = 10  # Number of tracks to recommend

    recommended_tracks = recommender.recommend_tracks(playlist_id=playlist_id, top_k=top_k)

    print(recommended_tracks)
    print(recommended_tracks[['track_uri', 'track_name', 'artist_name', 'predicted_rating']])


def evaluate() -> None:
    df_playlist_pure = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist_pure, min_tracks=100)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    evaluator = RecommenderEvaluator(df_playlist=df_playlist, df_tracks=df_tracks, type='hybrid')
    print(evaluator.evaluate())


def tune() -> None:
    df_playlist_pure = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist_pure, min_tracks=100)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    evaluator = RecommenderEvaluator(
        df_playlist=df_playlist,
        df_tracks=df_tracks,
        test_ratio=0.2,
        type='collaborative'
    )

    # === Step 1: Tune k for Collaborative Filtering ===
    k_values = [5, 20, 50, 100]
    k_hyperparameter_sets = [{'k': k} for k in k_values]

    k_results = evaluator.perform_k_fold_cv_parallel(
        hyperparameter_sets=k_hyperparameter_sets,
        n_splits=2,
    )

    print(k_results)

    best_k_result = max(k_results, key=lambda r: r['average_accuracy'])
    best_k = best_k_result['hyperparams']['k']

    print(f"Best k: {best_k} with Acc: {best_k_result['average_accuracy']:.3f}")

    # === Step 2: Tune Word2Vec Parameters with Best k ===
    evaluator = RecommenderEvaluator(
        df_playlist=df_playlist,
        df_tracks=df_tracks,
        test_ratio=0.2,
        type='content'
    )

    word2vec_params = {
        'D': [50, 100, 200],
        'w': [5, 10],
        'e': [10, 20, 30],
        'sg': [0, 1]
    }

    word2vec_hyperparameter_sets = [
        {'vector_size': D, 'window': w, 'epochs': e, 'sg': sg}
        for D, w, e, sg in product(*word2vec_params.values())
    ]

    word2vec_results = evaluator.perform_k_fold_cv_parallel(
        hyperparameter_sets=word2vec_hyperparameter_sets,
        n_splits=2
    )

    print(word2vec_results)

    best_word2vec_result = max(word2vec_results, key=lambda r: r['average_accuracy'])
    best_word2vec_params = best_word2vec_result['hyperparams']

    print(
        f"Best Word2Vec Params: {best_word2vec_params} with Acc: {best_word2vec_result['average_accuracy']:.3f}")

    # === Step 3: Tune Alpha for Hybrid Model ===
    evaluator = RecommenderEvaluator(
        df_playlist=df_playlist,
        df_tracks=df_tracks,
        test_ratio=0.2,
    )

    alpha_values = [0.2, 0.5, 0.8]
    alpha_hyperparameter_sets = [
        {**best_word2vec_params, 'content_weight': alpha}
        for alpha in alpha_values
    ]

    alpha_results = evaluator.perform_k_fold_cv_parallel(
        hyperparameter_sets=alpha_hyperparameter_sets,
        n_splits=2
    )

    print(alpha_results)

    best_alpha_result = max(alpha_results, key=lambda r: r['average_accuracy'])
    best_hyperparameters = best_alpha_result['hyperparams']

    print("Best Overall Hyperparameters:")
    print(best_hyperparameters)

    print(f"Final Accuracy: {best_alpha_result['average_accuracy']:.3f}")


if __name__ == "__main__":
    evaluate()
