from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, datasetname, seq_len=15, sample='evenly', transform=None, use_surf=False, candidate_len=15):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.candidate_len = candidate_len
        self.use_surf = use_surf
        self.datasetname = datasetname


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]

        num = len(img_paths)
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            if self.use_surf and len(frame_indices) >= self.candidate_len:
                # 随机得到一个end_index
                # 保证有一个self.candidate_len的长度
                rand_end = max(0, len(frame_indices) - self.candidate_len - 1)
                # 随机得到一个begin_index
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.candidate_len, len(frame_indices))

                indices = frame_indices[begin_index:end_index]
                for index in indices:
                    if len(indices) >= self.candidate_len:
                        break
                    indices.append(index)

                imgs_sift = list()
                sift2index = dict()
                for index in indices:
                    img_name = os.path.basename(img_paths[index])
                    txt_name = img_name.split('.')[0] + '.txt'
                    img_sift = np.loadtxt(img_paths[index].replace("DukeMTMC-VideoReID", "{}_surf_features".format(self.datasetname)).replace(img_name, txt_name), dtype=np.float32)
                    if len(img_sift) == 1:
                        img_sift = np.array([[0] * 64])
                    imgs_sift.append(img_sift)
                    sift2index[str(img_sift).strip()] = index

                imgs_sift = np.vstack(imgs_sift)

                # define criteria and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(imgs_sift, self.seq_len, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                # sift2index
                cluster = dict()
                for i in range(self.seq_len):
                    A = imgs_sift[label.ravel() == i]
                    cluster[i] = A
                    # cluster[i] = A.tolist()

                # 从每个簇中随机选出1个，一共4个index
                new_indices = list()
                for i in range(self.seq_len):
                    if len(cluster[i]) != 0:
                        rand_index = random.randint(0, len(cluster[i]) - 1)
                        # print(cluster[i][rand_index].shape)
                        new_index = sift2index[str(cluster[i][rand_index]).strip()]
                        new_indices.append(new_index)

                # 发生了簇的缺失的话就从余下的index中取得不重复的new_index
                needed_num = self.seq_len - len(new_indices)
                for i in range(needed_num):
                    new_indices.append((indices - new_indices)[0])

                indices = new_indices
                # ----------------------------------------------------------------

            else:
                rand_end = max(0, len(frame_indices) - self.seq_len - 1)
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.seq_len, len(frame_indices))

                indices = frame_indices[begin_index:end_index]

                for index in indices:
                    if len(indices) >= self.seq_len:
                        break
                    indices.append(index)
                indices=np.array(indices)

            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = list(range(num))
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))







