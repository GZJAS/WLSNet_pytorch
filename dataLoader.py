from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import os
import cv2
from glob import glob
from charSet import get_charSet, init_charSet

class videoDataset(Dataset):
    def __init__(self, path, videoMaxLen, txtMaxLen):
        self.queries = sorted(glob(os.path.join(path, '*.mp4')))
        self.labels = sorted(glob(os.path.join(path, '*.txt')))
        self.videoMaxLen = videoMaxLen
        self.txtMaxLen = txtMaxLen

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        video, length = videoProcess(self.queries[idx], self.videoMaxLen)
        return video, length, txtProcess(self.labels[idx], self.txtMaxLen)

def videoProcess(dir, videoMaxLen):
    cap = cv2.VideoCapture(dir)
    tmp = []
    results = torch.zeros(videoMaxLen, 120, 120)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(1, 120, 120)
            tmp.append(gray)
            # Display the resulting frame
            #imshow(gray, cmap='gray')
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    try:
        image_tensor = torch.from_numpy(np.concatenate(tmp, axis=0))
        idx = torch.tensor([i for i in range(image_tensor.size(0) - 1, -1, -1)])
        results[:image_tensor.size(0), :, :] = image_tensor[idx, :, :]
        length = torch.tensor(image_tensor.size(0)-4)
        if length <= 0:
            length = torch.tensor(1)
    except:
        print('#############################', dir)
        length = torch.tensor(1)
    cap.release()
    return results, length

def txtProcess(dir, txtMaxLen):

    tmp = []
    with open(dir) as f:
        tmp = [get_charSet().get_index_of(i) for i in f.readline().split(':')[1].replace(' ', '').rstrip('\n')] + [get_charSet().get_index_of('<eos>')]
        
        if len(tmp) < txtMaxLen:
            tmp += [get_charSet().get_index_of('<pad>') for _ in range(txtMaxLen - len(tmp))]
        
        else:
            print(len(tmp))
            raise Exception('too short txt max length')
    return torch.Tensor(tmp)

def get_dataloaders(path, batch_size, videomax, txtmax, num_worker, ratio_of_validation):
    num_workers = num_worker # num of threads to load data, default is 0. if you use thread(>1), don't confuse evenif debug messages are reported asynchronously.
    train_dataset = videoDataset(path, videomax, txtmax)
    num_train = len(train_dataset)
    split_point = int(ratio_of_validation*num_train)

    indices = list(range(num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split_point : ], indices[ : split_point]

    train_sampler = SubsetRandomSampler(train_idx) # Random sampling at every epoch without replacement in given indices
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
        pin_memory=True)
        # 'pin_memory=True' allows for you to use fast memory buffer with way of calling '.cuda(async=True)' function.

    val_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
        pin_memory=True)
    return [train_loader, val_loader]