import pandas as pd

from adv_ml_music_recommendation.recommenders.popularityrecommender import PopularityRecommender
from adv_ml_music_recommendation.recommenders.recommenderevaluator import RecommenderEvaluator


def prepare_data() -> pd.DataFrame:
    """
    This function might need to be changed in the future.
    Returns: A pandas dataframe containing playlist information
    """
    # Read main dataframe
    df = pd.read_csv("../data/1m.csv")
    df.rename(columns={'Unnamed: 0': 'playlist_id'}, inplace=True)

    # Additional artist features
    column_names = ['artist_uri', 'artist_popularity', 'genre']
    df_artist = pd.read_csv('../data/artist_features.csv', header=None, names=column_names)

    # Additional track features
    column_names = ['track_uri', 'release_date', 'track_popularity']

    # Read the CSV with column names
    df_track = pd.read_csv('../data/track_features.csv', header=None, names=column_names)

    # Merge df with df_artist on 'artist_uri'
    df_combined = pd.merge(df, df_artist, on='artist_uri', how='left')

    # Merge the resulting dataframe with df_track on 'track_uri'
    df_combined = pd.merge(df_combined, df_track, on='track_uri', how='left')

    return df_combined


def main() -> None:
    import mysql.connector as mysql
    import pandas as pd

    # Connect to the database using your custom user and password
    db = mysql.connect(
        host="localhost",  # MySQL host (localhost since the container is running locally)
        user="anon",  # Custom MySQL username
        password="spotify",  # Custom MySQL user's password
        database="spotifydb"  # The database to connect to
    )

    # Define the query
    query = "SELECT * FROM track LIMIT 10"

    # Fetch data into a Pandas DataFrame
    df = pd.read_sql(query, db)

    # Display the first few rows
    print(df.head())

    # Close the database connection
    db.close()
    """
    playlist_df = prepare_data()
    print(playlist_df.head())
    model_evaluator = RecommenderEvaluator(playlist_df)
    popularity_model = PopularityRecommender(playlist_df)
    popularity_model_metrics, popularity_model_details = model_evaluator.evaluate_model(popularity_model,                                                                       "popularity_model")

    print(popularity_model_metrics)
    print(popularity_model_details[[x for x in popularity_model_details.columns if x != 'playlist_id']] \
          .sort_values('recall@5', ascending=False) \
          .head(10))
    """


if __name__ == "__main__":
    main()
