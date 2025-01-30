from typing import Optional, List
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from adv_ml_music_recommendation.util.data_functions import get_tracks_by_playlist, filter_out_playlist_tracks


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
        self.attribute_list = attribute_list

        def discretize(value: float, threshold_low=0.4, threshold_high=0.7) -> str:
            if value <= threshold_low:
                return 'low'
            elif value <= threshold_high:
                return 'medium'
            else:
                return 'high'

        # Create the corpus based on the attributes, discretizing numeric values
        def create_track_sentence(track):
            sentence = []
            for attribute in self.attribute_list:
                value = track[attribute]
                if attribute == 'artists' and isinstance(value, str):  # Handle 'artists' as string representation of list
                    # Convert string representation of list to actual list
                    artists_list = ast.literal_eval(value)
                    # Append each artist as a separate token
                    sentence.extend(artists_list)
                elif isinstance(value, (int, float)):  # For numeric attributes
                    value = discretize(value)
                    sentence.append(f"{value}_{attribute}")
                    continue  # Skip the default string representation if it's numeric
                else:
                    sentence.append(str(value))  # For other string attributes
            return ' '.join(sentence)

        # Generate the corpus from the track data
        df_corpus = self.df_tracks.apply(create_track_sentence, axis=1)
        corpus = [row.split() for row in df_corpus]
        self.embedder = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, epochs=epochs, sg=sg)

        # Pre-compute track embeddings for all tracks
        self.track_embeddings = {
            track['track_uri']: self.construct_track_embedding(track)
            for _, track in self.df_tracks.iterrows()
        }

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

        # Filter out tracks already in the playlist
        candidate_tracks = filter_out_playlist_tracks(
            self.df_tracks,
            get_tracks_by_playlist(self.df_playlist, self.df_tracks, playlist_id)
        )

        # Retrieve precomputed embeddings for candidate tracks
        track_embeddings = np.vstack([
            self.track_embeddings[track['track_uri']]
            for _, track in candidate_tracks.iterrows() if track['track_uri'] in self.track_embeddings
        ])

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

    def get_closest_words(self, playlist_embedding: np.ndarray, top_n=10):
        """
        Finds the closest words in the Word2Vec space to the given embedding.

        :param playlist_embedding: The playlist embedding vector.
        :param top_n: Number of closest words to return.
        :return: List of (word, similarity score) tuples.
        """
        word_vectors = np.array([self.embedder.wv[word] for word in self.embedder.wv.index_to_key])
        similarities = cosine_similarity(playlist_embedding.reshape(1, -1), word_vectors)[0]

        word_similarities = list(zip(self.embedder.wv.index_to_key, similarities))
        sorted_words = sorted(word_similarities, key=lambda x: x[1], reverse=True)

        return sorted_words[:top_n]

    def visualize_embedding(self, playlist_embedding: np.ndarray, top_n=10):
        """
        Visualizes the closest words to the playlist embedding in 2D.

        :param playlist_embedding: The playlist embedding vector.
        :param top_n: Number of words to visualize.
        """
        closest_words = self.get_closest_words(playlist_embedding, top_n=top_n)
        words, similarities = zip(*closest_words)

        # Get word vectors for closest words
        word_vectors = np.array([self.embedder.wv[word] for word in words] + [playlist_embedding])

        dimensionality_reducer = PCA(n_components=2)

        dim_reduced_vectors = dimensionality_reducer.fit_transform(word_vectors)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(dim_reduced_vectors[:, 0], dim_reduced_vectors[:, 1], c='blue', label="Closest Words")

        # Highlight playlist embedding
        plt.scatter(dim_reduced_vectors[-1, 0], dim_reduced_vectors[-1, 1], c='red', marker='x', s=150, label="Playlist")

        # Annotate points
        for i, word in enumerate(words):
            plt.text(dim_reduced_vectors[i, 0], dim_reduced_vectors[i, 1], word, fontsize=12, ha='right', va='bottom')

        plt.text(dim_reduced_vectors[-1, 0], dim_reduced_vectors[-1, 1], "PLAYLIST", fontsize=14, fontweight='bold',
                 color='red', ha='right', va='bottom')

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"Closest Words to Playlist Embedding")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
