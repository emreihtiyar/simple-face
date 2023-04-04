import os
import requests
from simple_face.commons import DownloadConfig

def download_single_file(name, url, path="."):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path + name):
        print(f"{name} will be downloaded...")
        response = requests.get(url, stream=True, timeout=DownloadConfig.DOWNLOAD_TIMEOUT)
        with open(os.path.join(path, name), 'wb') as file:
            for chunk in response.iter_content(chunk_size=DownloadConfig.DOWNLOAD_CHUNK_SIZE):
                file.write(chunk)
        print(f"{name} has been downloaded.")
    return os.path.isfile(path + name)
