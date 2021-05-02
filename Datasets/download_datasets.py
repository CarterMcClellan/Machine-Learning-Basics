import requests
import os

from tqdm import tqdm

import torchtext

WAR_AND_PEACE = "https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt"

def download_from_url(url, filename):
    if os.path.exists(filename):
        print(filename, "already found")
        return
    
    r = requests.get(url)
    with open(filename, "wb") as f: 
        for data in tqdm(r.iter_content(), desc="Downloading: {}".format(filename)):
            f.write(data)

def download_IWSLT2016(target_dir='./IWSLT2016'):
    if os.path.exists(target_dir):
        print(target_dir, "already exists")
        return 
        
    torchtext.datasets.IWSLT2016(root=target_dir)
    print("Downloaded IWSLT2016")

if __name__ == "__main__":
    download_from_url(WAR_AND_PEACE, "war_and_peace.txt")
    download_IWSLT2016()
