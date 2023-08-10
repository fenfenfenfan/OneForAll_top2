# !/usr/bin/env python3
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import itertools
import random
from collections import defaultdict
from typing import Optional, List
import logging

import numpy as np
from paddle.io import Sampler
from utils import comm
logger = logging.getLogger(__name__)

def no_index(a, b):
    """no_index
    """
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


def reorder_index(batch_indices, world_size):
    r"""Reorder indices of samples to align with DataParallel training.
    In this order, each process will contain all images for one ID, triplet loss
    can be computed within each process, and BatchNorm will get a stable result.
    Args:
        batch_indices: A batched indices generated by sampler
        world_size: number of process
    Returns:

    """
    mini_batchsize = len(batch_indices) // world_size
    reorder_indices = []
    for i in range(0, mini_batchsize):
        for j in range(0, world_size):
            reorder_indices.append(batch_indices[i + j * mini_batchsize])
    return reorder_indices


# class BalancedIdentitySampler(Sampler):
#     def __init__(self, data_source, mini_batch_size, num_instances, seed=None):
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.num_pids_per_batch = mini_batch_size // self.num_instances

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()
#         self.batch_size = mini_batch_size * self._world_size

#         self.index_pid = dict()
#         self.pid_cam = defaultdict(list)
#         self.pid_index = defaultdict(list)

#         for index, info in enumerate(data_source):
#             pid = info[1]
#             camid = info[2]
#             self.index_pid[index] = pid
#             self.pid_cam[pid].append(camid)
#             self.pid_index[pid].append(index)

#         self.pids = sorted(list(self.pid_index.keys()))
#         self.num_identities = len(self.pids)

#         if seed is None:
#             seed = comm.shared_random_seed()
#         self._seed = int(seed)

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()

#     def __iter__(self):
#         start = self._rank
#         yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

#     def _infinite_indices(self):
#         np.random.seed(self._seed)
#         while True:
#             # Shuffle identity list
#             identities = np.random.permutation(self.num_identities)

#             # If remaining identities cannot be enough for a batch,
#             # just drop the remaining parts
#             drop_indices = self.num_identities % (self.num_pids_per_batch * self._world_size)
#             if drop_indices: identities = identities[:-drop_indices]

#             batch_indices = []
#             for kid in identities:
#                 i = np.random.choice(self.pid_index[self.pids[kid]])
#                 _, i_pid, i_cam = self.data_source[i]
#                 batch_indices.append(i)
#                 pid_i = self.index_pid[i]
#                 cams = self.pid_cam[pid_i]
#                 index = self.pid_index[pid_i]
#                 select_cams = no_index(cams, i_cam)

#                 if select_cams:
#                     if len(select_cams) >= self.num_instances:
#                         cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
#                     else:
#                         cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
#                     for kk in cam_indexes:
#                         batch_indices.append(index[kk])
#                 else:
#                     select_indexes = no_index(index, i)
#                     if not select_indexes:
#                         # Only one image for this identity
#                         ind_indexes = [0] * (self.num_instances - 1)
#                     elif len(select_indexes) >= self.num_instances:
#                         ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
#                     else:
#                         ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

#                     for kk in ind_indexes:
#                         batch_indices.append(index[kk])

#                 if len(batch_indices) == self.batch_size:
#                     yield from reorder_index(batch_indices, self._world_size)
#                     batch_indices = []


# class SetReWeightSampler(Sampler):
#     def __init__(self, data_source: str, mini_batch_size: int, num_instances: int, set_weight: list,
#                  seed: Optional[int] = None):
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.num_pids_per_batch = mini_batch_size // self.num_instances

#         self.set_weight = set_weight

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()
#         self.batch_size = mini_batch_size * self._world_size

#         assert self.batch_size % (sum(self.set_weight) * self.num_instances) == 0 and \
#                self.batch_size > sum(
#             self.set_weight) * self.num_instances, "Batch size must be divisible by the sum set weight"

#         self.index_pid = dict()
#         self.pid_cam = defaultdict(list)
#         self.pid_index = defaultdict(list)

#         self.cam_pid = defaultdict(list)

#         for index, info in enumerate(data_source):
#             pid = info[1]
#             camid = info[2]
#             self.index_pid[index] = pid
#             self.pid_cam[pid].append(camid)
#             self.pid_index[pid].append(index)
#             self.cam_pid[camid].append(pid)

#         # Get sampler prob for each cam
#         self.set_pid_prob = defaultdict(list)
#         for camid, pid_list in self.cam_pid.items():
#             index_per_pid = []
#             for pid in pid_list:
#                 index_per_pid.append(len(self.pid_index[pid]))
#             cam_image_number = sum(index_per_pid)
#             prob = [i / cam_image_number for i in index_per_pid]
#             self.set_pid_prob[camid] = prob

#         self.pids = sorted(list(self.pid_index.keys()))
#         self.num_identities = len(self.pids)

#         if seed is None:
#             seed = comm.shared_random_seed()
#         self._seed = int(seed)

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()

#     def __iter__(self):
#         start = self._rank
#         yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

#     def _infinite_indices(self):
#         np.random.seed(self._seed)
#         while True:
#             batch_indices = []
#             for camid in range(len(self.cam_pid.keys())):
#                 select_pids = np.random.choice(self.cam_pid[camid], size=self.set_weight[camid], replace=False,
#                                                p=self.set_pid_prob[camid])
#                 for pid in select_pids:
#                     index_list = self.pid_index[pid]
#                     if len(index_list) > self.num_instances:
#                         select_indexs = np.random.choice(index_list, size=self.num_instances, replace=False)
#                     else:
#                         select_indexs = np.random.choice(index_list, size=self.num_instances, replace=True)

#                     batch_indices += select_indexs
#             np.random.shuffle(batch_indices)

#             if len(batch_indices) == self.batch_size:
#                 yield from reorder_index(batch_indices, self._world_size)


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, mini_batch_size, num_instances, seed=None, dp_group=None, moe_group=None):
        data_source = dataset.img_items
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_pids_per_batch = mini_batch_size // self.num_instances

        # self._rank = comm.get_rank()
        # self._world_size = comm.get_world_size()
        if dp_group is None: 
            self._rank = comm.get_rank()
            self._world_size = comm.get_world_size()
        else:
            self._rank = comm.get_rank() // moe_group.nranks
            self._world_size = dp_group.nranks
        logger.info("dataset {}: rank {} is mapped to _rank {} under the real local world size {}".format(dataset.dataset_name, comm.get_rank(), self._rank, self._world_size))
        self.batch_size = mini_batch_size * self._world_size
        self.mini_batch_size = mini_batch_size
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        # self._generate_indices()
    
    # def _generate_indices(self):
    #     indices = self._finite_indices()
        
    #     reorder_indices = [] #TODO refine the process of reorder_indices
    #     for batch_indices in indices:
    #         for i in range(0, self.mini_batch_size):
    #              for j in range(0, self._world_size):
    #                 reorder_indices.append(batch_indices[i + j * self.mini_batch_size])
        
    #     local_indices = []
    #     for i in range(self._rank, len(reorder_indices), self._world_size):
    #         local_indices.append(reorder_indices[i])
        # self.local_indices = local_indices

    def __len__(self):
        return 0 #len(self.local_indices)

    def __iter__(self):
        # while True:
        #     for indice in self.local_indices:
        #         yield indice
        #     self._seed += 1
        #     self._generate_indices()
        # yield from self.local_indices
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    # def _finite_indices(self,):
    #     np.random.seed(self._seed)
    #     avl_pids = copy.deepcopy(self.pids)
    #     batch_idxs_dict = {}

    #     ret_indices = []
    #     batch_indices = []
    #     while len(avl_pids) >= self.num_pids_per_batch:
    #         selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
    #         for pid in selected_pids:
    #             # Register pid in batch_idxs_dict if not
    #             if pid not in batch_idxs_dict:
    #                 idxs = copy.deepcopy(self.pid_index[pid])
    #                 if len(idxs) < self.num_instances:
    #                     idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
    #                 np.random.shuffle(idxs)
    #                 batch_idxs_dict[pid] = idxs

    #             avl_idxs = batch_idxs_dict[pid]
    #             for _ in range(self.num_instances):
    #                 batch_indices.append(avl_idxs.pop(0))

    #             if len(avl_idxs) < self.num_instances: avl_pids.remove(pid)

    #         if len(batch_indices) == self.batch_size:
    #             ret_indices.append( batch_indices)
    #             batch_indices = []
    #     return ret_indices
    
    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            avl_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}

            batch_indices = []
            while len(avl_pids) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
                for pid in selected_pids:
                    # Register pid in batch_idxs_dict if not
                    if pid not in batch_idxs_dict:
                        idxs = copy.deepcopy(self.pid_index[pid])
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                        np.random.shuffle(idxs)
                        batch_idxs_dict[pid] = idxs

                    avl_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avl_idxs.pop(0))

                    if len(avl_idxs) < self.num_instances: avl_pids.remove(pid)

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self._world_size)
                    batch_indices = []


# class NaiveIdentitySamplerFaster(Sampler):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#     Args:
#     - data_source (list): list of (img_path, pid, camid).
#     - num_instances (int): number of instances per identity in a batch.
#     - batch_size (int): number of examples in a batch.
#     """

#     def __init__(self, data_source: str, mini_batch_size: int, num_instances: int, seed: Optional[int] = None):
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.num_pids_per_batch = mini_batch_size // self.num_instances

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()
#         self.batch_size = mini_batch_size * self._world_size

#         self.pid_index = defaultdict(list)

#         for index, info in enumerate(data_source):
#             pid = info[1]
#             self.pid_index[pid].append(index)

#         self.pids = sorted(list(self.pid_index.keys()))
#         self.num_identities = len(self.pids)

#         if seed is None:
#             seed = comm.shared_random_seed()
#         self._seed = int(seed)

#     def __iter__(self):
#         start = self._rank
#         yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

#     def _infinite_indices(self):
#         np.random.seed(self._seed)
#         while True:
#             avl_pids = self.pids
#             # avl_pids = copy.deepcopy(self.pids)

#             batch_indices = []
#             while len(avl_pids) >= self.num_pids_per_batch:
#                 selected_pids = random.choices(avl_pids, k=self.num_pids_per_batch)
#                 # selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
#                 for pid in selected_pids:
#                     # idxs = copy.deepcopy(self.pid_index[pid])
#                     idxs = self.pid_index[pid]
#                     # if len(idxs) < self.num_instances:
#                     #     idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()

#                     # selected_idxs = np.random.choice(idxs, self.num_instances, replace=False).tolist()
#                     selected_idxs = random.choices(idxs, k=self.num_instances)

#                     batch_indices.extend(selected_idxs)

#                 if len(batch_indices) == self.batch_size:
#                     yield from reorder_index(batch_indices, self._world_size)
#                     batch_indices = []
