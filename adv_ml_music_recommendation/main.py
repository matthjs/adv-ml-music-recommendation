import pandas as pd
from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender
from adv_ml_music_recommendation.recommenders.hybridrecommender import HybridRecommender
from adv_ml_music_recommendation.recommenders.recommenderevaluator import RecommenderEvaluator
from adv_ml_music_recommendation.util.data_functions import filter_playlist_df_min_tracks


def main() -> None:
    df_playlist = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist, min_tracks=69)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    recommender = CollaborativeRecommender(df_playlist=df_playlist, df_tracks=df_tracks)

    # Get the top 10 recommended tracks for a specific playlist
    playlist_id = 33660  # Replace with the playlist_id for which you want recommendations
    top_k = 10  # Number of tracks to recommend

    recommended_tracks = recommender.recommend_tracks(playlist_id=playlist_id, top_k=top_k)

    print(recommended_tracks)
    print(recommended_tracks[['track_uri', 'track_name', 'artist_name', 'predicted_rating']])


def evaluate() -> None:
    df_playlist_pure = pd.read_csv("../data/track_playlist_association.csv")
    df_playlist = filter_playlist_df_min_tracks(df_playlist_pure, min_tracks=69)
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    evaluator = RecommenderEvaluator(df_playlist=df_playlist, df_tracks=df_tracks, type='collaborative')
    print(evaluator.evaluate())


if __name__ == "__main__":
    evaluate()
