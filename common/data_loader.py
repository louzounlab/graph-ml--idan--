import os
import torch
import pickle
import networkx as nx
from collections import OrderedDict

import numpy as np
from scipy import sparse

from sklearn.model_selection import train_test_split

from features_infra.feature_calculators import z_scoring
from features_infra.graph_features import GraphFeatures, get_max_subgraph

from loggers import EmptyLogger

DTYPE = np.float32


def symmetric_normalization(mx):
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum != 0] **= -0.5
    r_inv = rowsum.flatten()
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)  # D^-0.5 * X * D^-0.5


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def handle_matrix_concat(mx, should_normalize=True):
    mx += sparse.eye(mx.shape[0])
    mx_t = mx.transpose()
    if should_normalize:
        mx = symmetric_normalization(mx)
        mx_t = symmetric_normalization(mx_t)

    return sparse.vstack([mx, mx_t])  # vstack: below, hstack: near


def handle_matrix_symmetric(mx, should_normalize=True):
    # build symmetric adjacency matrix
    mx += (mx.T - mx).multiply(mx.T > mx)
    mx += sparse.eye(mx.shape[0])
    return symmetric_normalization(mx) if should_normalize else mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(DTYPE)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def activate_cuda(*items, cuda_num=None):
    if cuda_num is None:
        return items[0] if 1 == len(items) else items
    if 1 == len(items):
        return items[0].cuda(cuda_num)
    return [x.cuda(cuda_num) for x in items]


class GraphLoader(object):
    def __init__(self, paths, is_max_connected, ignore_index=-1, norm_adj=True, logger=None, cuda_num=None, dtype=torch.double):
        super(GraphLoader, self).__init__()
        self._logger = EmptyLogger() if logger is None else logger
        self._paths = paths
        self._ignore_index = ignore_index
        self._cuda_num = cuda_num
        self._dtype = dtype

        self._logger.debug("Loading %s dataset...", paths["features"])
        self._gnx = pickle.load(open(paths["gnx"], "rb"))
        self._is_max_connected = is_max_connected
        if is_max_connected:
            self._gnx = get_max_subgraph(self._gnx)

        self.ordered_nodes = sorted(self._gnx)
        self._labeled_nodes = set(i for i, n in enumerate(self.ordered_nodes) if "label" in self._gnx.node[n])
        # self._labeled_nodes = [(i, n) for i, n in enumerate(self.ordered_nodes) if "label" in self._gnx.node[n]]
        self._labels = {i: label for i, label in enumerate(self._gnx.graph["node_labels"])}
        self._node_labels = self._get_node_labels()

        self._content = OrderedDict(sorted(pickle.load(open(paths["content"], "rb")).items(), key=lambda x: x[0]))
        bow_mx = np.vstack(self._content.values()).astype(DTYPE)
        median_bow = np.median(bow_mx, axis=0)
        bow_mx = np.vstack([self._content.get(node, median_bow) for node in self.ordered_nodes]).astype(DTYPE)

        self._bow_mx = z_scoring(bow_mx)
        self._topo_mx = None

        # Adjacency matrices
        adj = nx.adjacency_matrix(self._gnx, nodelist=self.ordered_nodes).astype(DTYPE)
        self._adj = handle_matrix_symmetric(adj, should_normalize=norm_adj)
        self._adj = sparse_mx_to_torch_sparse_tensor(self._adj).to_dense()
        self._adj_rt = handle_matrix_concat(adj, should_normalize=norm_adj)
        self._adj_rt = sparse_mx_to_torch_sparse_tensor(self._adj_rt).to_dense()

        self._train_set = self._test_set = None
        self._train_idx = self._test_idx = self._base_train_idx = None
        self._val_idx = None

    @property
    def name(self):
        return str(self._paths["name"])

    @property
    def is_graph_directed(self):
        return self._gnx.is_directed()

    # def _activate_cuda(self, items):
        # return items

    # def _encode_onehot_gnx(self):  # gnx, nodes_order: list = None):
    #     labels = self._labels.copy()
    #     if labels[len(labels) - 1] is not None:
    #         labels[len(labels)] = None
    #     ident = np.identity(len(labels))
    #     if self._gnx.graph.get('is_index_labels', False):
    #         labels_dict = {i: ident[i, :] for i, label in labels.items()}
    #     else:
    #         labels_dict = {label: ident[i, :] for i, label in labels.items()}
    #     return np.array(list(map(lambda n: labels_dict[self._gnx.node[n].get('label')], self._nodes_order)),
    #                     dtype=np.int32)

    def _get_node_labels(self):
        labels = self._labels.copy()
        labels[self._ignore_index] = None
        labels_dict = {label: i for i, label in labels.items()}
        return np.array(list(map(lambda n: labels_dict[self._gnx.node[n].get('label')], self.ordered_nodes)),
                        dtype=np.int32)

    def set_variables(self, **kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @property
    def num_labels(self):
        return len(self._labels)

    # @property
    # def labels(self):
    #     labels = torch.LongTensor(self._node_labels)
    #     return activate_cuda(labels, cuda_num=self._cuda_num)

    @property
    def labels(self):
        labels = torch.LongTensor(self._node_labels)
        return activate_cuda(labels, cuda_num=self._cuda_num)

    @property
    def distinct_labels(self):
        return sorted(self._labels.keys())

    # def _get_idx(self, idx_name):
    #     return torch.LongTensor([x for x in getattr(self, idx_name) if x in set(self._labeled_nodes)])
    #
    # @property
    # def train_idx(self):
    #     return activate_cuda(self._get_idx("_train_idx"), cuda_num=self._cuda_num)
    #
    # @property
    # def val_idx(self):
    #     return activate_cuda(self._get_idx("_val_idx"), cuda_num=self._cuda_num)
    #
    # @property
    # def test_idx(self):
    #     return activate_cuda(self._get_idx("_test_idx"), cuda_num=self._cuda_num)

    @property
    def bow_mx(self):
        # bow_feat = torch.FloatTensor(self._bow_mx)
        bow_feat = torch.DoubleTensor(self._bow_mx)
        return activate_cuda(bow_feat, cuda_num=self._cuda_num)

    @property
    def topo_mx(self):
        assert self._topo_mx is not None, "Split train required"
        # topo_feat = torch.FloatTensor(self._topo_mx)
        topo_feat = torch.DoubleTensor(self._topo_mx)
        return activate_cuda(topo_feat, cuda_num=self._cuda_num)

    @property
    def adj_rt_mx(self):
        return activate_cuda(self._adj_rt, cuda_num=self._cuda_num)  # .clone())

    @property
    def adj_mx(self):
        return activate_cuda(self._adj, cuda_num=self._cuda_num).type(self._dtype)  # .clone())

        # def split_test(self, test_p):
        #     indexes, nodes = zip(*self._labeled_nodes)
        #     self._train_set, _, self._base_train_idx, self._test_idx = train_test_split(nodes, indexes, test_size=test_p)

        # def split_train(self, train_p, features_meta):
        #     train_set, val_set, self._train_idx, self._val_idx = train_test_split(self._train_set, self._base_train_idx,
        #                                                                           test_size=1 - train_p)
        #     feat_path = os.path.join(self._feat_path, "features%d" % (self._is_max_connected,))
        # features = GraphFeatures(self._gnx, features_meta, dir_path=self._paths["features"], logger=self._logger,
        #                          is_max_connected=False)  # Already taking the max sub_graph in init
        # features.build(include=set(train_set), should_dump=False)
        #
        # add_ones = bool({"first_neighbor_histogram", "second_neighbor_histogram"}.intersection(features_meta))
        # self._topo_mx = features.to_matrix(add_ones=add_ones, dtype=np.float64, mtype=np.matrix, should_zscore=True)
        #
        # ratio = 10 ** np.ceil(np.log10(abs(np.mean(self._topo_mx) / np.mean(self._bow_mx))))
        # self._topo_mx /= ratio

    def set_train(self, train_set, features_meta):
        features = GraphFeatures(self._gnx, features_meta, dir_path=self._paths["features"], logger=self._logger,
                                 is_max_connected=False)  # Already taking the max sub_graph in init
        features.build(include=set(train_set), should_dump=True)

        add_ones = bool({"first_neighbor_histogram", "second_neighbor_histogram"}.intersection(features_meta))
        self._topo_mx = features.to_matrix(add_ones=add_ones, dtype=np.float64, mtype=np.matrix, should_zscore=True)

        ratio = 10 ** np.ceil(np.log10(abs(np.mean(self._topo_mx) / np.mean(self._bow_mx))))
        self._topo_mx /= ratio


class MultiGraphLoader(OrderedDict):
    def __init__(self, path_info, is_max_connected, *args, logger=None, cuda_num=None, is_debug=False, **kwargs):
        super(MultiGraphLoader, self).__init__()
        # def __init__(self, path_info, is_max_connected, norm_adj=True, cuda_num=None, logger=None):
        # path_info = {"years": os.path.realpath(os.path.join(PROJ_DIR, "..", "data", "firms", "years")),
        #              "label": "top"}
        self._path_info = path_info
        self._path_info["split"] = os.path.realpath(os.path.join(self._path_info["years"], "..", "split.pkl"))

        if logger is None:
            logger = EmptyLogger()
        self._logger = logger

        # TODO: implement dynamic loading of the data
        for year in sorted(os.listdir(path_info["years"]), key=int):
            data_path = os.path.realpath(os.path.join(path_info["years"], year))
            year_paths = {
                "features": os.path.join(data_path, "features%d" % (is_max_connected,)),
                "content": os.path.join(data_path, "content_clean.pkl"),
                # "content": os.path.join(data_path, path_info["label"], "content.pkl"),
                "gnx": os.path.join(data_path, path_info["label"], "gnx.pkl"),
                "name": str(year),
            }
            self[int(year)] = GraphLoader(year_paths, is_max_connected, *args, logger=self._logger, cuda_num=cuda_num, **kwargs)
            if is_debug and (1997 == int(year)): break

        self._nodes = np.array(self.ordered_nodes)  # getattr will take the first one
        self._test_idx = self._base_train_idx = None
        self._train_idx = self._val_idx = None
        self._should_split = not os.path.exists(self._path_info["split"])
        if not self._should_split:
            self._load_split()

    # def _get_idx(self, idx_name):
    #     return torch.LongTensor([x for x in getattr(self, idx_name) if x in set(self._labeled_nodes)])

    # @property
    def train_idx(self):
        train_idx = torch.LongTensor(self._train_idx)
        return activate_cuda(train_idx, cuda_num=self._cuda_num)

    # @property
    def val_idx(self):
        val_idx = torch.LongTensor(self._val_idx)
        return activate_cuda(val_idx, cuda_num=self._cuda_num)

    # @property
    def test_idx(self):
        test_idx = torch.LongTensor(self._test_idx)
        return activate_cuda(test_idx, cuda_num=self._cuda_num)

    # @property
    # def train_idx(self):
        # return activate_cuda(self._train_idx, cuda_num=self._cuda_num)
        # return self._train_idx

    # @property
    # def val_idx(self):
        # return activate_cuda(self._val_idx, cuda_num=self._cuda_num)
        # return self._val_idx

    # @property
    # def test_idx(self):
        # return activate_cuda(self._test_idx, cuda_num=self._cuda_num)
        # return self._test_idx

    @property
    def indexes(self):
        return self.train_idx, self.val_idx, self.test_idx

    @property
    def labels(self):
        return torch.stack([loader.labels for loader in self], dim=1)

    def split_test(self, test_p):
        test_p /= 100.
        if self._should_split:
            indexes = list(range(len(self._nodes)))
            self._base_train_idx, self._test_idx = train_test_split(indexes, test_size=test_p)
            self._test_idx = np.array(self._test_idx)
            self._base_train_idx = np.array(self._base_train_idx)

        # for loader in self:
        #     loader.set_variables(_test_idx=self._test_idx)

    def _load_split(self):
        r = pickle.load(open(self._path_info["split"], "rb"))
        self._base_train_idx, self._test_idx = r["base"], r["test"]
        self._train_idx, self._val_idx = r["train"], r["val"]

    def split_train(self, train_p, feature_meta):
        train_p *= (1. + (float(len(self._test_idx)) / len(self._base_train_idx))) / 100.

        if self._should_split:
            self._train_idx, self._val_idx = train_test_split(self._base_train_idx, test_size=1 - train_p)
            pickle.dump(
                {"train": self._train_idx, "val": self._val_idx, "test": self._test_idx, "base": self._base_train_idx},
                open(self._path_info["split"], "wb"))

        for loader in self:
            loader.set_train(set(self._nodes[self._train_idx]), feature_meta)
            # loader.set_variables(_train_idx=self._test_idx, _val_idx=self._val_idx)

    def __iter__(self):
        return iter(self.values())

    def __getattr__(self, item):
        return getattr(next(iter(self)), item)
        # return lambda gnx_idx: self[gnx_idx].__getattribute__(item)

    def __getitem__(self, key):
        if key in self or not (isinstance(key, int) and 0 <= key < len(self)):
            return super(MultiGraphLoader, self).__getitem__(key)
        return list(self.values())[key]
