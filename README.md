<br />
<p align="center">
  <h1 align="center">Music Recommendation System for Spotify
Playlists</h1>

  <p align="center">
  </p>
</p>

## About The Project
A variety of machine learning paradigms underpin recommender systems. In this project we focus on a hybrid recommendation system that uses model-based collaborative filtering, involving the singular value decomposition (SVD) and content-based filtering based on word2vec word embeddings. 

The recommendation task at hand is playlist continuation, whereby for each playlist we aim to suggest the top k most relevant additional tracks.
## Getting started

### Prerequisites
- [Docker v4.25](https://www.docker.com/get-started) or higher (if running docker container).
- [Poetry](https://python-poetry.org/).
## Running

Using docker: Run the docker-compose files to run all relevant services (`docker compose up` or `docker compose up --build`).

You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

# License
This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.
