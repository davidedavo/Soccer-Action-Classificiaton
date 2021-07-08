import numpy as np
import downloader as dl

def gather_data():
    password = "s0cc3rn3t"
    dl.downloadDataset(password)

if __name__ == "__main__":
    gather_data()