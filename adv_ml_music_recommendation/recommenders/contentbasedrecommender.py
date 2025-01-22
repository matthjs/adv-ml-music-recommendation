from typing import Optional, List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from adv_ml_music_recommendation.util.data_functions import get_tracks_by_playlist


class ContentRecommender(AbstractSongRecommender):
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame,
                 attribute_list: Optional[List[str]] = None,
                 vector_size: int = 100,
                 window: int = 5,
                 epochs: int = 5,
                 sg: int = 0):
        """
        :param df_playlist: DataFrame containing 'track_uri' and 'playlist_id'.
        :param df_tracks: DataFrame containing detailed track information.
        :param attribute_list: What song features/attributes one wants to use for computing word embeddings.
        :param vector_size: Size of the word embedding vectors [Word2Vec hyperparameter].
        :param window: Maximum distance between the current and predicted word within a sentence [Word2Vec hyperparameter].
        :param sg: Training algorithm: 1 for skip-gram; otherwise CBOW [Word2Vec hyperparameter].
        """
        super().__init__(df_playlist, df_tracks)
        if attribute_list is None:
            attribute_list = ['name', 'artists', 'danceability', 'energy', 'loudness',
                              'tempo', 'release_date', 'acousticness', 'speechiness']
        # Artists becomes "['Artist_1', "Artist_2']"
        self.attribute_list = attribute_list

        # On class instantiation immediately "fits" to data. Something we want to change? Idk.
        df_corpus = self.df_tracks[attribute_list].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        # print(df_corpus[0:10])
        corpus = [row.split() for row in df_corpus]
        # print(corpus[0:10])
        self.embedder = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, epochs=epochs, sg=sg)
        # Pre-compute track embeddings of all tracks
        self.track_embeddings = []
        for _, track in self.df_tracks.iterrows():  # Is there a more efficient way than to use a for loop?
            track_embedding = self.construct_track_embedding(track)
            self.track_embeddings.append(track_embedding)

    def get_average_w2v(self, tokens: List[str], vector_size: int = 100) -> np.ndarray:
        """
        Get the average Word2Vec vector for a list of tokens (words).

        :param tokens: List of tokens (words).
        :param vector_size: Dimensionality of the embeddings.
        :return: Average Word2Vec vector for the list of tokens.
        """
        valid_vectors = [self.embedder.wv[word] for word in tokens if word in self.embedder.wv]
        if valid_vectors:
            return np.mean(valid_vectors, axis=0)
        else:
            return np.zeros(vector_size)

    def construct_playlist_embedding(self, playlist_id: int) -> np.ndarray:
        """
        Constructs the embedding for a given playlist based on the average of its tracks' embeddings.

        :param playlist_id: ID of the playlist.
        :return: Playlist embedding (average of the track embeddings).
        """
        playlist_tracks = get_tracks_by_playlist(self.df_playlist, self.df_tracks, playlist_id)

        # Get the embeddings for each track in the playlist
        track_embeddings = []
        for _, track in playlist_tracks.iterrows():
            embedding = self.construct_track_embedding(track)
            track_embeddings.append(embedding)

        playlist_embedding = np.mean(track_embeddings, axis=0)
        return playlist_embedding

    def construct_track_embedding(self, track: pd.Series) -> np.ndarray:
        track_tokens = track[self.attribute_list].astype(str).values.tolist()  # Text features for Word2Vec
        embedding = self.get_average_w2v(track_tokens)
        return embedding

    def recommend_tracks(self, playlist_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommends top N tracks for the given playlist based on semantic similarity (cosine similarity).

        :param playlist_id: ID of the playlist for which recommendations are made.
        :param top_k: Number of recommended tracks.
        :return: DataFrame containing the top K recommended tracks.
        """
        playlist_embedding = self.construct_playlist_embedding(playlist_id)

        track_embeddings = np.vstack(self.track_embeddings)
        similarities = cosine_similarity(playlist_embedding.reshape(1, -1), track_embeddings).flatten()

        # Normalize cosine similarity to the range [-1,1] --> [0, 1]
        normalized_similarities = (similarities + 1) / 2  # Shift to [0, 2] and then scale to [0, 1]

        # Get the indices of the top K similar tracks
        top_indices = normalized_similarities.argsort()[-top_k:][::-1]
        recommended_tracks = self.df_tracks.loc[top_indices]
        # Add predicted ratings
        predicted_ratings = normalized_similarities[top_indices]
        recommended_tracks['predicted_rating'] = predicted_ratings

        return recommended_tracks
