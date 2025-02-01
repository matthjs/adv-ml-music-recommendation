from typing import Optional, List
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from adv_ml_music_recommendation.recommenders.abstractrecommender import AbstractSongRecommender
from adv_ml_music_recommendation.util.data_functions import get_tracks_by_playlist
from gensim.models import Word2Vec


class ContentRecommender(AbstractSongRecommender):
    def __init__(self, df_playlist: pd.DataFrame, df_tracks: pd.DataFrame,
                 attribute_list: Optional[List[str]] = None,
                 vector_size: int = 100,
                 window: int = 15,
                 epochs: int = 30,
                 sg: int = 1,
                 cached_content_recommender=None):
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
            # TODO: Update this to include more attributes!
            attribute_list = ['name', 'artists', 'danceability', 'energy',
                              'tempo', 'release_date', 'acousticness', 'speechiness',
                              'instrumentalness',
                              'liveness',
                              # 'loudness',
                              # 'valence'
                              ]
        self.attribute_list = attribute_list

        if cached_content_recommender is None:
            # Generate corpus using consistent tokenization
            df_corpus = self.df_tracks.apply(self.get_track_tokens, axis=1)
            corpus = [' '.join(tokens).split() for tokens in df_corpus]  # Split into final tokens
            self.embedder = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, epochs=epochs, sg=sg)
            # Precompute track embeddings using correct tokenization

            # Create index mapping
            self.track_index = {uri: idx for idx, uri in enumerate(self.df_tracks['track_uri'])}

            # Precompute all embeddings in vectorized form
            self.embedding_matrix = np.array([
                self.construct_track_embedding(track)
                for _, track in self.df_tracks.iterrows()
            ])
        else:
            self.embedder = cached_content_recommender.embedder
            self.track_index = cached_content_recommender.track_index
            self.embedding_matrix = cached_content_recommender.embedding_matrix

    def discretize(self, value: float, threshold_low=0.4, threshold_high=0.7) -> str:
        if value <= threshold_low:
            return 'low'
        elif value <= threshold_high:
            return 'medium'
        else:
            return 'high'

    def get_track_tokens(self, track: pd.Series) -> List[str]:
        tokens = []
        for attribute in self.attribute_list:
            value = track[attribute]
            if attribute == 'artists':
                if isinstance(value, str):
                    artists = ast.literal_eval(value)
                else:
                    artists = value
                tokens.extend(artists)
            elif isinstance(value, (int, float)):
                disc_value = self.discretize(value)
                tokens.append(f"{disc_value}_{attribute}")
            else:
                tokens.extend(str(value).split())
        return tokens

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
        indices = [self.track_index[uri] for uri in playlist_tracks['track_uri']]
        return self.embedding_matrix[indices].mean(axis=0)

    def construct_track_embedding(self, track: pd.Series) -> np.ndarray:
        track_tokens = self.get_track_tokens(track)
        return self.get_average_w2v(track_tokens, self.embedder.vector_size)

    def recommend_tracks(self, playlist_id: int, top_k: int = 10) -> pd.DataFrame:
        playlist_embedding = self.construct_playlist_embedding(playlist_id)
        # Get candidate tracks not in the playlist
        candidate_mask = ~self.df_tracks['track_uri'].isin(
            get_tracks_by_playlist(self.df_playlist, self.df_tracks, playlist_id)['track_uri']
        )

        # Vectorized cosine similarity
        similarities = cosine_similarity(
            playlist_embedding.reshape(1, -1),
            self.embedding_matrix[candidate_mask]
        ).flatten()

        probabilities = softmax(similarities)

        if top_k < 0:
            return (
                self.df_tracks[candidate_mask]
                .assign(predicted_rating=probabilities)
                .sort_values('predicted_rating', ascending=False)
            )

        # Get top-K indices
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

        return (
            self.df_tracks[candidate_mask]
            .iloc[top_indices]
            .assign(predicted_rating=probabilities[top_indices])
        )

    def visualize_embedding(self, playlist_embedding: np.ndarray, top_n: int = 10):
        """
        Visualizes the closest tracks to the playlist embedding in 2D PCA space,
        displaying song names and printing track details.
        """
        # Get all track URIs and their indices
        track_uris = self.df_tracks['track_uri'].values
        track_indices = list(self.track_index.values())

        # Calculate similarities
        similarities = cosine_similarity(
            playlist_embedding.reshape(1, -1),
            self.embedding_matrix
        ).flatten()

        # Get top indices
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        top_track_uris = track_uris[top_indices]

        # Retrieve track details from DataFrame
        top_tracks_info = self.df_tracks[self.df_tracks['track_uri'].isin(top_track_uris)]
        top_tracks_info = top_tracks_info.set_index('track_uri').loc[top_track_uris]  # Maintain order

        # Print track information
        print(f"\n=== Top {top_n} Closest Tracks ===")
        for idx, (uri, track) in enumerate(zip(top_track_uris, top_tracks_info.itertuples()), 1):
            # Parse artists from string representation
            artists = ast.literal_eval(track.artists) if isinstance(track.artists, str) else track.artists
            print(f"\n#{idx}: {track.name}")
            print(f"  Artists: {', '.join(artists)}")
            print(f"  Album: {track.album_name}")
            print(f"  Track URI: {uri}")

        # Prepare data for visualization
        vectors_to_plot = np.vstack([
            self.embedding_matrix[top_indices],
            playlist_embedding
        ])

        pca = TSNE(n_components=2, perplexity=10)
        reduced_vectors = pca.fit_transform(vectors_to_plot)

        plt.figure(figsize=(8, 6))

        # Annotate points
        plt.scatter(
            reduced_vectors[:-1, 0],
            reduced_vectors[:-1, 1],
            c='blue',
            label='Recommended Tracks'
        )

        # Highlight playlist embedding
        plt.scatter(
            reduced_vectors[-1, 0],
            reduced_vectors[-1, 1],
            c='red',
            marker='X',
            s=200,
            label='Playlist'
        )

        # Annotate track names
        for i, (x, y) in enumerate(reduced_vectors[:-1]):
            plt.annotate(
                top_tracks_info.iloc[i]['name'],
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=9,
                alpha=0.8
            )

        # Annotate playlist
        plt.annotate(
            "PLAYLIST",
            (reduced_vectors[-1, 0], reduced_vectors[-1, 1]),
            textcoords="offset points",
            xytext=(0, 15),
            ha='center',
            fontsize=12,
            fontweight='bold',
            color='red'
        )

        # Add plot metadata
        plt.xlabel("T-SNE Dim 1")
        plt.ylabel("T-SNE Dim 2")
        plt.title(f"Top {top_n} Tracks Closest to Playlist Embedding")
        plt.legend()
        plt.grid(linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
