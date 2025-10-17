# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import os
import torch
import copy
import numpy as np
from PIL import Image
from tabulate import tabulate
from termcolor import colored
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms as T

class ReidEvaluator:
    def __init__(self, num_query, output_dir=None):
        self._predictions = []
        self._num_query = num_query

    def process(self, inputs, outputs):
        prediction = {
            'feats': torch.from_numpy(outputs),
            'pids': inputs['targets'],
            'camids': inputs['camids']
        }

        self._predictions.append(prediction)
    
    def compute_cosine_distance(self, features, others):
        """Computes cosine distance.
        Args:
            features (torch.Tensor): 2-D feature matrix.
            others (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        features = F.normalize(features, p=2, dim=1)
        others = F.normalize(others, p=2, dim=1)
        dist_m = 1 - torch.mm(features, others.t())
        return dist_m.cpu().numpy()
    
    def evaluate_rank(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
        num_q, num_g = distmat.shape

        if num_g < max_rank:
            max_rank = num_g
            print('Note: number of gallery samples is quite small, got {}'.format(num_g))

        indices = np.argsort(distmat, axis=1)
        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        all_INP = []
        num_valid_q = 0.  # number of valid query

        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            matches = (g_pids[order] == q_pid).astype(np.int32)
            raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()

            pos_idx = np.where(raw_cmc == 1)
            max_pos_idx = np.max(pos_idx)
            inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
            all_INP.append(inp)

            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q

        return all_cmc, all_AP, all_INP
        
    def evaluate(self):
        predictions = self._predictions

        features = []
        pids = []
        camids = []

        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
        
        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()

        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]

        self._results = OrderedDict()

        dist = self.compute_cosine_distance(query_features, gallery_features)
        
        cmc, all_AP, all_INP = self.evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        return copy.deepcopy(self._results)
    
class VehicleID(torch.utils.data.Dataset):
    def __init__(self, root='datasets', test_list="", image_size=(256, 256)):
        self.image_dir = os.path.join(root, "image")

        if test_list:
            self.test_list = test_list
        else:
            self.test_list = os.path.join(root, 'train_test_split/test_list_13164.txt')
        
        required_files = [
            root,
            self.image_dir,
            self.test_list
        ]
        
        self.check_before_run(required_files)
        self.query, self.gallery = self.process_dir(self.test_list)

        self.transforms = T.Compose([
            T.Resize(image_size, interpolation=3),
            T.ToTensor(),
        ])

        self.img_items = self.query + self.gallery

        pid_set = set()
        cam_set = set()

        for i in self.img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
        
        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = Image.open(img_path)
        img = self.transforms(img) * 255.0

        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path
        }
    
    def __len__(self):
        return len(self.img_items)

    def check_before_run(self, required_files):
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))
    
    def process_dir(self, list_file):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for line in img_list_lines:
            line = line.strip()
            vid = int(line.split(' ')[1])
            img_id = line.split(' ')[0]
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            img_id = int(img_id)
            dataset.append((img_path, vid, img_id))
        
        # random.shuffle(dataset)
        vid_container = set()
        query = []
        gallery = []
        for sample in dataset:
            if sample[1] not in vid_container:
                vid_container.add(sample[1])
                gallery.append(sample)
            else:
                query.append(sample)
        
        return query, gallery

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
        return len(pids), len(cams)

    def show_test(self):
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [
            ['query', num_query_pids, len(self.query), num_query_cams],
            ['gallery', num_gallery_pids, len(self.gallery), num_gallery_cams],
        ]

        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        print(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
    
class SmallVehicleID(VehicleID):
    """VehicleID.
    Small test dataset statistics:
        - identities: 800.
        - images: 6493.
    """
    def __init__(self, root='datasets'):
        self.test_list = os.path.join(root, 'train_test_split/test_list_800.txt')
        super(SmallVehicleID, self).__init__(root, self.test_list)

class MediumVehicleID(VehicleID):
    """VehicleID.
    Medium test dataset statistics:
        - identities: 1600.
        - images: 13377.
    """
    def __init__(self, root='datasets'):
        self.test_list = os.path.join(root, 'train_test_split/test_list_1600.txt')
        super(MediumVehicleID, self).__init__(root, self.test_list)

class LargeVehicleID(VehicleID):
    """VehicleID.
    Large test dataset statistics:
        - identities: 2400.
        - images: 19777.
    """
    
    def __init__(self, root='datasets'):
        self.test_list = os.path.join(root, 'train_test_split/test_list_2400.txt')
        super(LargeVehicleID, self).__init__(root, self.test_list)