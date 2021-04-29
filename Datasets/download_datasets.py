import requests
import os

from tqdm import tqdm

WAR_AND_PEACE = "https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt"

def download_from_url(url, filename):
    if os.path.exists(filename):
        print(filename, "already found")
        return
    
    r = requests.get(url)
    with open(filename, "wb") as f: 
        for data in tqdm(r.iter_content(), desc="Downloading: {}".format(filename)):
            f.write(data)

if __name__ == "__main__":
    download_from_url(WAR_AND_PEACE, "war_and_peace.txt")