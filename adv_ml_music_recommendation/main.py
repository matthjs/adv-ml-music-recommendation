import pandas as pd
from adv_ml_music_recommendation.recommenders.hybridrecommender import HybridRecommender
from adv_ml_music_recommendation.recommenders.recommenderevaluator import RecommenderEvaluator
from adv_ml_music_recommendation.util.data_functions import filter_playlist_df_min_tracks, get_tracks_by_playlist
from itertools import product


def main() -> None:
    # Shows a recommendation example
    df_playlist = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist, min_tracks=10)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    # Get the top 10 recommended tracks for a specific playlist
    playlist_id = 3  # Replace with the playlist_id for which you want recommendations
    top_k = 5  # Number of tracks to recommend

    playlist_tracks = get_tracks_by_playlist(df_playlist, df_tracks, playlist_id)
    print(playlist_tracks)

    recommender = HybridRecommender(df_playlist=df_playlist, df_tracks=df_tracks)

    recommended_tracks = recommender.recommend_tracks(playlist_id=playlist_id, top_k=top_k)

    print(recommended_tracks)


def evaluate() -> None:
    df_playlist_pure = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist_pure, min_tracks=50)
    print(len(df_playlist['playlist_id'].unique()))
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    evaluator = RecommenderEvaluator(df_playlist=df_playlist, df_tracks=df_tracks)

    print(evaluator.evaluate())

    evaluator2 = RecommenderEvaluator(df_playlist=df_playlist, df_tracks=df_tracks, type='collaborative')
    evaluator3 = RecommenderEvaluator(df_playlist=df_playlist, df_tracks=df_tracks, type='content')

    print(evaluator2.evaluate())
    print(evaluator3.evaluate())


def tune() -> None:
    # Hyperparameter tuning experiment
    df_playlist_pure = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist_pure, min_tracks=50)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    # File paths for saving results
    collaborative_results_file = "../results/collaborative_results.csv"
    word2vec_results_file = "../results/word2vec_results.csv"
    hybrid_results_file = "../results/hybrid_results.csv"

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

    # Save collaborative filtering results
    print(k_results)
    pd.DataFrame(k_results).to_csv(collaborative_results_file, index=False)

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
        'w': [5, 10, 15],
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

    # Save content-based filtering results
    print(word2vec_results)
    pd.DataFrame(word2vec_results).to_csv(word2vec_results_file, index=False)

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
        {'k': best_k, **best_word2vec_params, 'content_weight': alpha}
        for alpha in alpha_values
    ]

    alpha_results = evaluator.perform_k_fold_cv_parallel(
        hyperparameter_sets=alpha_hyperparameter_sets,
        n_splits=2
    )

    # Save hybrid model results
    print(alpha_results)
    pd.DataFrame(alpha_results).to_csv(hybrid_results_file, index=False)

    best_alpha_result = max(alpha_results, key=lambda r: r['average_accuracy'])
    best_hyperparameters = best_alpha_result['hyperparams']

    print("Best Overall Hyperparameters:")
    print(best_hyperparameters)

    print(f"Final Accuracy: {best_alpha_result['average_accuracy']:.3f}")


if __name__ == "__main__":
    main()
