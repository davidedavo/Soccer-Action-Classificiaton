import os
import numpy as np
import torch

def npToTorch(path, dataset_path):
    id = 0
    champs = os.listdir(path)
    for c in champs:
        c_path = path + "/" + c
        years = os.listdir(c_path)
        for y in years:
            game_tensor = []
            y_path = c_path + "/" + y
            games = os.listdir(y_path)
            for g in games:
                g_path = y_path + "/" + g
                ts_game = torch.from_numpy(np.load(f"{g_path}/features.npz")['arr_0'])
                game_tensor += ts_game
                np_labels = np.load(f"{g_path}/labels.npz")['arr_0']
                np_labels = np.concatenate(np_labels)
            game_tensor = torch.stack(game_tensor)
            torch.save(game_tensor, f"{dataset_path}/{id}.pt")
            id += 1
