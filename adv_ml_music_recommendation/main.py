import pandas as pd
from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender
from adv_ml_music_recommendation.recommenders.contentbasedrecommender import ContentRecommender


def main() -> None:
    df_playlist = pd.read_csv("../data/track_playlist_association.csv")
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    recommender = ContentRecommender(df_playlist=df_playlist, df_tracks=df_tracks)

    # Get the top 10 recommended tracks for a specific playlist
    playlist_id = 12345  # Replace with the playlist_id for which you want recommendations
    top_k = 10  # Number of tracks to recommend

    recommended_tracks = recommender.recommend_tracks(playlist_id=playlist_id, top_k=top_k)

    print(recommended_tracks)
    print(recommended_tracks[['track_uri', 'track_name', 'artist_name', 'predicted_rating']])

    recommender2 = CollaborativeRecommender(df_playlist=df_playlist, df_tracks=df_tracks)

    # Get the top 10 recommended tracks for a specific playlist
    playlist_id = 12345  # Replace with the playlist_id for which you want recommendations
    top_k = 10  # Number of tracks to recommend

    recommended_tracks2 = recommender2.recommend_tracks(playlist_id=playlist_id, top_k=top_k)

    print(recommended_tracks2)
    print(recommended_tracks2[['track_uri', 'track_name', 'artist_name', 'predicted_rating']])


if __name__ == "__main__":
    main()
