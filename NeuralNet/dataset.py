import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt
import json

def downloadDataset():
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=".data")

    # download labels SN v2
    mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
    # download labels for camera shot
    mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) 
    
    mySoccerNetDownloader.password = input("Password for videos?:\n")
    # download LQ Videos
    mySoccerNetDownloader.downloadGames(files=["1.mkv", "2.mkv"], split=["train","valid","test","challenge"])


def readVideo(championshipName = "italy_serie-a", year = "2014-2015",baseDir = ".data"):
    path = f"{baseDir}/{championshipName}/{year}"
    folderList = os.listdir(path)
    dataset = []
    for f in folderList:
        
        labelpath = path+f"/{f}/Labels-v2.json"

        with open(labelpath) as lab:
            data = json.load(lab)
            annot = data["annotations"]
            goals = [d for d in annot if d["label"]=="Goal"]
            print(goals)
            for goal in goals:
                gameTime = goal["gameTime"]
                time = gameTime.split("-")[0].strip()
                minutes = gameTime.split("-")[1].split(":")[0].strip()
                seconds = gameTime.split("-")[1].split(":")[1].strip()
                seconds = int(seconds) + int(minutes)*60

                videopath = path+f"/{f}/{time}.mkv"
                with VideoFileClip(videopath) as video:
                    new = video.subclip(seconds - 20, seconds + 10)
                    frames = []
                    for frame in new.iter_frames(fps=1):
                        frames.append(frame)
                        #plt.imshow(frame)
                        #plt.pause(0.5)
                dataset.append((frames, "Goal"))
    return dataset

if __name__ == "__main__":
    data = readVideo()
    print(data)
    print(len(data))

