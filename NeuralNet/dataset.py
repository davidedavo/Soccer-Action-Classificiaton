import numpy as np
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt
import json
import time

def downloadDataset():
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=".data/clear")

    # download labels SN v2
    mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
    # download labels for camera shot
    mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) 
    
    mySoccerNetDownloader.password = input("Password for videos?:\n")
    # download LQ Videos
    mySoccerNetDownloader.downloadGames(files=["1.mkv", "2.mkv"], split=["train","valid","test","challenge"])


def readVideo(championships = None, years = None, baseDir = ".data/clear",  excludeLabels = ["Ball out of play", "Kick-off", "Throw-in", "Substitution"]):
    """read from video and return a list of frames for each video and the corresponding labels

    Args:
        championships ([type], optional): Filters the championships. If None there is no filter. Defaults to None.
        years ([type], optional): Filters the years. If None there is no filter.. Defaults to None.
        baseDir (str, optional): Defaults to ".data/clear".
        excludeLabels (list, optional): List of labels to exclude. Defaults to ["Ball out of play", "Kick-off", "Throw-in", "Substitution"].

    Returns:
        np.ndarray: Shape (n_actions, frames_per_action, width, height, channels),
        np.ndarray: Shape (n_actions),
    """
    championships = championships if championships is not None else os.listdir(baseDir)
    X = []
    Y = []
    for championshipName in championships:
        years = years if years is not None else os.listdir(f"{baseDir}/{championshipName}")
        for year in years:
            path = f"{baseDir}/{championshipName}/{year}"
            folderList = os.listdir(path)
            for f in folderList:
                labelpath = path+f"/{f}/Labels-v2.json"
                print(f"partita {f}")
                with open(labelpath) as lab:
                    data = json.load(lab)
                    annot = data["annotations"]
                    actions = [d for d in annot if d["label"] == "Goal"]
                    for act in actions:
                        gameTime = act["gameTime"]
                        time = gameTime.split("-")[0].strip()
                        minutes = gameTime.split("-")[1].split(":")[0].strip()
                        seconds = gameTime.split("-")[1].split(":")[1].strip()
                        seconds = int(seconds) + int(minutes)*60

                        videopath = path+f"/{f}/{time}.mkv"
                        if not os.path.isfile(videopath):
                            continue 
                        try:
                            with VideoFileClip(videopath) as video:
                                #numframes = 20
                                new = video.subclip(seconds - 10, seconds + 10)
                                frames = []
                                for frame in new.iter_frames(fps=1):
                                    frames.append(frame)
                                    #plt.imshow(frame)
                                    #plt.pause(0.1)
                            X.append(frames)
                            Y.append("Goal")
                        except Exception:
                            pass
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    return X,Y

def save_compressed(X,Y):
    t = round(time.time() * 1000)
    np.savez_compressed(f".data/compressed/X_{t}.npz", X)
    np.savez_compressed(f".data/compressed/Y_{t}.npz", Y)

if __name__ == "__main__":
    X, Y = readVideo()
    save_compressed(X,Y)

