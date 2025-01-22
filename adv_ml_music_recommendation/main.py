import pandas as pd
from adv_ml_music_recommendation.recommenders.collaborativerecommender import CollaborativeRecommender


def main() -> None:
    df_playlist = pd.read_csv("../data/track_playlist_association.csv")
    df_tracks = pd.read_csv("../data/matched_songs.csv")

    recommender = CollaborativeRecommender(df_playlist=df_playlist, df_tracks=df_tracks, k=15)

    # Get the top 10 recommended tracks for a specific playlist
    playlist_id = 12345  # Replace with the playlist_id for which you want recommendations
    top_k = 10  # Number of tracks to recommend

    recommended_tracks = recommender.recommend_tracks(playlist_id=playlist_id, top_k=top_k)

    print(recommended_tracks)
    print(recommended_tracks[['track_uri', 'track_name', 'artist_name', 'predicted_rating']])


if __name__ == "__main__":
    main()
