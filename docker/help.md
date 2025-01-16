You want
docker
  - dumps
    - spotifydbdumpschemashare.sql   # need to be downloaded locally
    - spotifydbdumpshare.sql
  - docker-compose.yml

 Then run `docker compose up -d` to setup the database

 A script like
```
import pandas as pd
from sqlalchemy import create_engine

# Define the connection string
connection_string = "mysql+mysqlconnector://anon:spotify@host.docker.internal:3306/spotifydb"

# Create an SQLAlchemy engine
engine = create_engine(connection_string)

# Define the query
query = "SELECT * FROM track LIMIT 10"

# Fetch data into a Pandas DataFrame
df = pd.read_sql(query, engine)

# Display the first few rows
print(df.head())

# Close the engine connection (optional, since SQLAlchemy handles connections)
engine.dispose()
 ```