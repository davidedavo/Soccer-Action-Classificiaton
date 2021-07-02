import torch
import torchvision
import cv2
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader



def downloadDataset():
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=".data")

    # download labels SN v2
    mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
    # download labels for camera shot
    mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) 
    
    mySoccerNetDownloader.password = input("Password for videos?:\n")
    # download LQ Videos
    mySoccerNetDownloader.downloadGames(files=["1.mkv", "2.mkv"], split=["train","valid","test","challenge"])