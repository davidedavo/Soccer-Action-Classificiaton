import os
import numpy as np
import torch
from sklearn import preprocessing as pp

def npToTorch(path, dataset_path):
    # LabelEncoder Initialization
    label_encoder = pp.LabelEncoder()

    id = 0
    champs = os.listdir(path)
    desired_shape = torch.Size([224, 398, 3])
    for c in champs:
        print(f"Processing {c}...")
        c_path = path + "/" + c
        years = os.listdir(c_path)
        for y in years:
            print(f"Year: {y}")
            game_tensor = []
            game_label = []
            y_path = c_path + "/" + y
            games = os.listdir(y_path)
            for g in games:
                print(f"Game: {g}")
                g_path = y_path + "/" + g

                ts_game = torch.from_numpy(np.load(f"{g_path}/features.npz")['arr_0'])
                if ts_game.shape[-3:] != desired_shape:
                    continue 

                np_labels = np.load(f"{g_path}/labels.npz")['arr_0']
                classes, count = np.unique(np_labels, return_counts=True)
                if 'Goal' not in classes:
                    continue
                idx = np.where(classes == 'Goal')
                n_goals = count[idx]
                print(f"Goals: {n_goals}")
                d = {
                    'Goal': 0,
                    'Foul': 0,
                    'Corner': 0,
                    'noAction': 0
                }
                for i, act in enumerate(np_labels):
                    if all(val == n_goals for val in d.values()):
                        break
                    if d[act] < n_goals:
                        game_tensor.append(ts_game[i])
                        game_label.append(np_labels[i])
                        d[act] += 1
                
                print(f"Actions saved: {d}")

            game_label = np.array(game_label)

            print(f"Year {y} completed, compressing...")
            # Storing a tensor for each year with related encoded labels
            game_tensor = torch.stack(game_tensor)
            torch.save(game_tensor, f"{dataset_path}/{id}.pt")

            encoded_labels = torch.as_tensor(label_encoder.fit_transform(game_label))
            torch.save(encoded_labels, f"{dataset_path}/{id}_lab.pt")
            id += 1
            print(f"Done! Final Shape: {game_tensor.shape}")

if __name__ == "__main__":
    npToTorch(".data", ".dataset")
