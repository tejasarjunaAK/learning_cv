from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile, os

# Authenticate
api = KaggleApi()
api.authenticate()

# Paths
competition_name = "dogs-vs-cats-redux-kernels-edition"
download_path = "dogs-vs-cats-redux"
os.makedirs(download_path, exist_ok=True)

# Download competition files
api.competition_download_files(competition_name, path=download_path, quiet=False)

# Extract zip files
for file in os.listdir(download_path):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(download_path, file), 'r') as zip_ref:
            zip_ref.extractall(download_path)

print("Download and extraction complete!")
