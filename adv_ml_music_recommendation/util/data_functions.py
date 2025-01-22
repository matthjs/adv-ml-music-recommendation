import numpy as np
import pandas as pd
from typing import Tuple
import pandas as pd
from scipy.sparse import csr_array


def get_interacted_tracks(
        df_playlist: pd.DataFrame,
        df_tracks: pd.DataFrame,
        playlist_id: int,
        drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the track dataset into tracks associated with a specific playlist
    and tracks not associated with that playlist.

    :param df_playlist: DataFrame containing 'track_uri' and 'playlist_id'.
    :param df_tracks: DataFrame containing detailed track information.
    :param playlist_id: The playlist ID for which to identify tracks.
    :param drop_duplicates: Whether to remove duplicate track URIs. Defaults to True.
    :return: A tuple (tracks_interacted, tracks_not_interacted).
    """
    # Get track URIs in the specified playlist
    interacted_track_ids = set(df_playlist[df_playlist['playlist_id'] == playlist_id]['track_uri'])

    # Tracks associated with the playlist
    tracks_interacted = df_tracks[df_tracks['track_uri'].isin(interacted_track_ids)]

    # Tracks not associated with the playlist
    tracks_not_interacted = df_tracks[~df_tracks['track_uri'].isin(interacted_track_ids)]

    # Drop duplicates if specified
    if drop_duplicates:
        tracks_interacted = tracks_interacted.drop_duplicates(subset='track_uri', keep='first').reset_index(drop=True)
        tracks_not_interacted = tracks_not_interacted.drop_duplicates(subset='track_uri', keep='first').reset_index(
            drop=True)

    return tracks_interacted, tracks_not_interacted


def get_tracks_by_playlist_associate(df: pd.DataFrame, playlist_id: int) -> list:
    """
    Get all track URIs for a specific playlist ID.

    :param df: The DataFrame containing 'track_uri' and 'playlist_id'.
    :param playlist_id: The playlist ID to filter tracks for.
    :return: A list of track URIs associated with the playlist.
    """
    return df[df['playlist_id'] == playlist_id]['track_uri']


def get_tracks_by_playlist(df_playlist: pd.DataFrame, df_tracks: pd.DataFrame, playlist_id: int) -> pd.DataFrame:
    """
    Get detailed track information for a specific playlist ID.

    :param df_playlist: DataFrame containing 'track_uri' and 'playlist_id'.
    :param df_tracks: DataFrame containing detailed track information.
    :param playlist_id: The playlist ID to filter tracks for.
    :return: A DataFrame with detailed track information for the playlist.
    """
    # Filter the playlist DataFrame for the specified playlist ID
    filtered_playlist = df_playlist[df_playlist['playlist_id'] == playlist_id]

    # Merge with the track information DataFrame on 'track_uri'
    return filtered_playlist.merge(df_tracks, on='track_uri', how='inner')


def get_number_of_playlists(df: pd.DataFrame) -> int:
    """
    Get the number of unique playlists.

    :param df: The DataFrame containing playlist information.
    :return: The number of unique playlist IDs.
    """
    return df['playlist_id'].nunique()


def get_number_of_songs_in_playlist(df: pd.DataFrame, playlist_id: int) -> int:
    """
    Get the number of songs in a specific playlist.

    :param df: The DataFrame containing 'track_uri' and 'playlist_id'.
    :param playlist_id: The playlist ID to count songs for.
    :return: The number of songs in the playlist.
    """
    return df[df['playlist_id'] == playlist_id].shape[0]


def create_interaction_matrix(df_playlist: pd.DataFrame, df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a user-item interaction matrix where rows represent playlists and columns represent tracks.

    :param df_playlist: DataFrame containing 'track_uri' and 'playlist_id'.
    :param df_tracks: DataFrame containing detailed track information including 'track_uri' and 'id'.
    :return: DataFrame representing the user-item interaction matrix.
    """
    # Step 1: Merge df_playlist and df_tracks on 'track_uri' to get track 'id'
    merged_df = pd.merge(df_playlist, df_tracks[['track_uri', 'id']], on='track_uri', how='left')

    # Step 2: Create a column 'event_strength' to represent interaction (1 for interaction)
    merged_df['event_strength'] = 1

    # Step 3: Create a user-item interaction matrix using pivot_table
    interaction_matrix_df = merged_df.pivot_table(
        index='playlist_id',  # Rows: playlist_id
        columns='id',  # Columns: track_id (id from df_tracks)
        values='event_strength',  # Values: Interaction strength
        aggfunc='sum',  # Aggregate by summing (since event_strength is 1)
    ).fillna(0)  # Fill NaN with 0 to indicate no interaction

    return interaction_matrix_df


def create_sparse_interaction_matrix(df_playlist: pd.DataFrame, df_tracks: pd.DataFrame) -> csr_array:
    """
    Creates a sparse user-item interaction matrix where rows represent playlists and columns represent tracks.

    :param df_playlist: DataFrame containing 'track_uri' and 'playlist_id'.
    :param df_tracks: DataFrame containing detailed track information including 'track_uri' and 'id'.
    :return: Sparse matrix representing the user-item interaction matrix.
    """
    # Step 1: Merge df_playlist and df_tracks on 'track_uri' to get 'track_id'
    merged_df = pd.merge(df_playlist, df_tracks[['track_uri', 'id']], on='track_uri', how='left')

    # Step 2: Create a column 'event_strength' to represent interaction (1 for interaction)
    merged_df['event_strength'] = 1

    # Step 3: Create mappings for playlist_id and track_id to integer indices
    playlist_id_mapping = {playlist_id: idx for idx, playlist_id in enumerate(merged_df['playlist_id'].unique())}
    track_id_mapping = {track_id: idx for idx, track_id in enumerate(merged_df['id'].unique())}

    # Step 4: Convert 'playlist_id' and 'id' to integer indices
    merged_df['playlist_index'] = merged_df['playlist_id'].map(playlist_id_mapping)
    merged_df['track_index'] = merged_df['id'].map(track_id_mapping)

    # Step 5: Create a sparse user-item interaction matrix
    rows = merged_df['playlist_index'].values
    cols = merged_df['track_index'].values
    data = merged_df['event_strength'].values

    # Create a sparse matrix (CSR format)
    interaction_matrix_sparse = csr_array((data, (rows, cols)), shape=(len(playlist_id_mapping), len(track_id_mapping)),
                                          dtype=np.float64)

    return interaction_matrix_sparse


def filter_out_playlist_tracks(recommendations: pd.DataFrame, playlist_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out tracks that are already in the given playlist from the recommendation list.

    :param recommendations: DataFrame containing recommended tracks.
    :param playlist_tracks: DataFrame containing tracks already in the playlist.
    :return: DataFrame containing recommended tracks with tracks already in the playlist removed.
    """
    # Get the track URIs already in the playlist
    playlist_track_uris = playlist_tracks['track_uri'].values

    # Filter out tracks already in the playlist
    return recommendations[~recommendations['track_uri'].isin(playlist_track_uris)]
