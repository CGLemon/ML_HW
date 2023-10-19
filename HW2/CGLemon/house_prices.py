import pandas as pd
import glob, os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FullyConnect(nn.Module):
    def __init__(self, in_size,
                       out_size,
                       act=nn.Identity):
        super(FullyConnect, self).__init__()
        self.act = act
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        o = self.linear(x)
        return self.act(o)

class Model(nn.Module):
    def __init__(self, in_size, out_size):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            FullyConnect(in_size, 512, act=nn.ReLU()),
            FullyConnect(512, 1024, act=nn.ReLU()),
            # FullyConnect(1024, 2048, act=nn.ReLU()),
            # FullyConnect(2048, 1024, act=nn.ReLU()),
            FullyConnect(1024, 512, act=nn.ReLU()),
            FullyConnect(512, out_size, act=nn.ReLU())
        )

    def forward(self, x):
        o = self.layers(x)
        return o

class MyDataset(Dataset):
    def __init__(self, x_chunk, y_chunk):
        self.x_chunk = x_chunk
        self.y_chunk = y_chunk

    def __len__(self):
        return len(self.x_chunk)

    def __getitem__(self, idx):
        x, y = self.x_chunk[idx], self.y_chunk[idx]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y

def load_all_csv():
    train_name = os.path.join("kaggle", "train.csv")
    test_name = os.path.join("kaggle", "test.csv")
    train_df = pd.read_csv(train_name)
    test_df = pd.read_csv(test_name)
    return train_df, test_df

def print_info(info):
    for i, result in enumerate(info):
        label = result["label"]
        items = result["items"]
        idx = i+1

        out = str()
        if result["is_obj"]:
            out += "{}. {} -> {}".format(idx, label, items)
        else:
            l = items[0]
            h = items[-1]
            out += "{}. {} -> {}~{}".format(idx, label,l,h)
        if result["NaN"]:
            out += ", include {} NaN...".format(result["NaN"])
        print(out)

def get_data_format(info):
    dformat = dict()

    for result in info:
        label = result["label"]
        if label == "Id":
            continue

        data = result["data"]
        items = result["items"]
        if result["is_obj"]:
            vmap = dict()
            for i, v in enumerate(items):
                vmap[v] = i
            vmap["NaN"] = len(items)
            dformat[label] = {
                "type" : "one-hot",
                "size" : len(items) + 1,
                "map" : vmap
            }
        else:
            val_accm = 0
            cnt = 0
            for k, v in data.items():
                val_accm += (k * v)
                cnt += v
            mean = val_accm/cnt
            dformat[label] = {
                "type" : "scalar",
                "mean" : mean
            }
    return dformat

def transfer_data(df, dformat):
    labels = df.keys().values
    x_chunk = list()
    y_chunk = list()
    for index, row in df.iterrows():
        x = list()
        y = list()
        for label in labels:
            if label == "Id":
                continue
            val = row[label]
            fmt = dformat[label]

            if fmt["type"] == "one-hot":
                if not type(val) == str and math.isnan(val):
                    val = "NaN"
                sub = [0] * fmt["size"]
                idx = fmt["map"][val]
                sub[idx] = 1
            else:
                if label == "SalePrice":
                    sub = [val/fmt["mean"]]
                else:
                    sub = [0] * 2
                    if math.isnan(val):
                        sub[0] = 1.
                    else:
                        sub[1] = val/fmt["mean"]

            if label == "SalePrice":
                y.extend(sub)
            else: 
                x.extend(sub)
        x_chunk.append(x)
        y_chunk.append(y)
    x_chunk = np.array(x_chunk, dtype=np.float32)
    y_chunk = np.array(y_chunk, dtype=np.float32)
    return x_chunk, y_chunk

def process_df_info(df):
    labels = df.keys().values
    info = list()

    for label in labels:
        result = {}
        data_dict = {}
        num_nan = 0

        for v in df[label]:
            if not type(v) == str and math.isnan(v):
                num_nan += 1
                continue

            is_obj = type(v) == str
            if not v in data_dict:
                data_dict[v] = 1
            else:
                data_dict[v] += 1

        result["label"] = label
        result["items"] = sorted(list(data_dict.keys()))
        result["data"] = data_dict
        result["NaN"] = num_nan
        result["is_obj"] = is_obj
        info.append(result)
    return info

def split(x_chunk, y_chunk, r=0.9):
    size = round(len(y_chunk) * r) 
    train_x = x_chunk[:size]
    train_y = y_chunk[:size]
    test_x = x_chunk[size:]
    test_y = y_chunk[size:]
    return train_x, train_y, test_x, test_y

def test_performance(model, loader, device):
    err, cnt = 0, 0
    for idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        predict = model(x)

        val = torch.mean(torch.abs(predict - y)).item()
        err += val
        cnt += 1
    print(err/cnt)

def print_results(model, loader, device, dformat):
    pred_result = "Id,SalePrice\n"
    for idx, batch in enumerate(loader):
        x, _ = batch
        x = x.to(device)
        predict = model(x)
        mean = dformat["SalePrice"]["mean"]
        price = mean * predict.item()
        pred_result += "{},{:.4f}\n".format(idx+1461, price)
    pred_result = pred_result[:-1]
    print(pred_result)

def run(use_gpu, read):
    train_df, pred_df = load_all_csv()

    dformat = get_data_format(process_df_info(train_df))
    x_chunk, y_chunk = transfer_data(train_df, dformat)
    in_size = x_chunk.shape[1]
    train_x, train_y, test_x, test_y = split(x_chunk, y_chunk, r=0.9)
    model = Model(in_size, 1)

    device = torch.device("cpu")
    if use_gpu:
        device = torch.device("cuda")
        model = model.to(device)

    if not read:
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        MSE = nn.MSELoss()
        loss_accm = 0

        dataset = MyDataset(train_x, train_y)
        loader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=True)

        max_steps = 10000
        num_steps = 0
        running = True

        model.train()
        while running:
            for idx, batch in enumerate(loader):
                x, y = batch
                x, y = x.to(device), y.to(device)
                predict = model(x)
                loss = MSE(predict, y)

                loss.backward()
                opt.step()
                opt.zero_grad()

                num_steps += 1
                loss_accm += loss.item()
                if num_steps % 100 == 0:
                    print("{} -> loss: {}".format(num_steps, loss_accm/100))
                    loss_accm = 0
                if num_steps >= max_steps:
                    running = False
                    break
        torch.save(model.to(torch.device("cpu")).state_dict(), "house_model.pt")
        model = model.to(device)
    else:
        model = model.to(torch.device("cpu"))
        model.load_state_dict(torch.load("house_model.pt"))
        model = model.to(device)

    model.eval()
    dataset = MyDataset(test_x, test_y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=True)
    test_performance(model, loader, device)

    x_chunk, y_chunk = transfer_data(pred_df, dformat)
    dataset = MyDataset(x_chunk, y_chunk)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    print_results(model, loader, device, dformat)

if __name__ == "__main__":
    run(True, True)
