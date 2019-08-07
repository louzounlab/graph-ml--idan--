# from __future__ import division
# from __future__ import print_function

import argparse
import logging
import pickle
import random
import time
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable

import model_meter
from feature_meta import NODE_FEATURES, DIRECTED_NEIGHBOR_FEATURES, UNDIRECTED_NEIGHBOR_FEATURES
from gcn import *
from gcn.data_loader import GraphLoader
from gcn.layers import AsymmetricGCN
from gcn.models import GCNCombined, GCN
from loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger


def get_features(feat_type, is_directed):
    neighbor_features = DIRECTED_NEIGHBOR_FEATURES if is_directed else UNDIRECTED_NEIGHBOR_FEATURES
    all_features = {"neighbors": [neighbor_features],
                    "features": [NODE_FEATURES],
                    "combined": [neighbor_features, NODE_FEATURES],
                    }

    return dict(y for x in all_features[feat_type] for y in x.items())


class ModelRunner:
    def __init__(self, products_path, dataset_path, conf, logger, data_logger=None):
        self.conf = conf
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self.products_path = products_path

        self.loader = GraphLoader(dataset_path, is_max_connected=False, norm_adj=conf["norm_adj"],
                                  cuda_num=conf["cuda"], logger=self._logger)

        self._criterion = torch.nn.NLLLoss()

    def _get_models(self):
        bow_feat = self.loader.bow_mx
        topo_feat = self.loader.topo_mx

        model1 = GCN(nfeat=bow_feat.shape[1],
                     hlayers=[self.conf["kipf"]["hidden"]],
                     nclass=self.loader.num_labels,
                     dropout=self.conf["kipf"]["dropout"])
        opt1 = optim.Adam(model1.parameters(), lr=self.conf["kipf"]["lr"],
                          weight_decay=self.conf["kipf"]["weight_decay"])

        model2 = GCNCombined(nbow=bow_feat.shape[1],
                             nfeat=topo_feat.shape[1],
                             hlayers=self.conf["hidden_layers"],
                             nclass=self.loader.num_labels,
                             dropout=self.conf["dropout"])
        opt2 = optim.Adam(model2.parameters(), lr=self.conf["lr"], weight_decay=self.conf["weight_decay"])

        model3 = GCN(nfeat=topo_feat.shape[1],
                     hlayers=self.conf["multi_hidden_layers"],
                     nclass=self.loader.num_labels,
                     dropout=self.conf["dropout"],
                     layer_type=None)
        opt3 = optim.Adam(model3.parameters(), lr=self.conf["lr"], weight_decay=self.conf["weight_decay"])

        model4 = GCN(nfeat=topo_feat.shape[1],
                     hlayers=self.conf["multi_hidden_layers"],
                     nclass=self.loader.num_labels,
                     dropout=self.conf["dropout"],
                     layer_type=AsymmetricGCN)
        opt4 = optim.Adam(model4.parameters(), lr=self.conf["lr"], weight_decay=self.conf["weight_decay"])

        return {
            "kipf": {
                "model": model1, "optimizer": opt1,
                "arguments": [self.loader.bow_mx, self.loader.adj_mx],
                "labels": self.loader.labels,
            },
            "our_combined": {
                "model": model2, "optimizer": opt2,
                "arguments": [self.loader.bow_mx, self.loader.topo_mx, self.loader.adj_rt_mx],
                "labels": self.loader.labels,
            },
            "topo_sym": {
                "model": model3, "optimizer": opt3,
                "arguments": [self.loader.topo_mx, self.loader.adj_mx],
                "labels": self.loader.labels,
            },
            "topo_asym": {
                "model": model4, "optimizer": opt4,
                "arguments": [self.loader.topo_mx, self.loader.adj_rt_mx],
                "labels": self.loader.labels,
            },
        }

    def run(self, train_p, feat_type):
        features_meta = get_features(feat_type, is_directed=self.loader.is_graph_directed)
        self.loader.split_train(train_p, features_meta)

        models = self._get_models()

        if self.conf["cuda"] is not None:
            [model["model"].cuda(self.conf["cuda"]) for model in models.values()]

        for model in models.values():
            model["arguments"] = list(map(Variable, model["arguments"]))
            model["labels"] = Variable(model["labels"])

        # Train model
        meters = {name: model_meter.ModelMeter(self.loader.distinct_labels) for name in models}
        train_idx, val_idx = self.loader.train_idx, self.loader.val_idx
        for epoch in range(self.conf["epochs"]):
            for name, model_args in models.items():
                self._train(epoch, name, model_args, train_idx, val_idx, meters[name])

        # Testing
        test_idx = self.loader.test_idx
        for name, model_args in models.items():
            meter = meters[name]
            self._test(name, model_args, test_idx, meter)
            self._data_logger.log_info(
                model_name=name,
                loss=meter.last_val("loss_test"),
                acc=meter.last_val("acc_test"),
                train_p=(train_p / (2 - train_p)) * 100,
                norm_adj=self.conf["norm_adj"],
                feat_type=self.conf["feat_type"]
            )

            # Currently supporting only binary class plotting
            # meters[name].plot_auc(should_show=False)
            # import matplotlib.pyplot as plt
            # plt.savefig(os.path.join(self.products_path, time.strftime("%H_%M_%S_" + name)))

        return meters

    def _train(self, epoch, model_name, model_args, idx_train, idx_val, meter):
        model, optimizer = model_args["model"], model_args["optimizer"]
        arguments, labels = model_args["arguments"], model_args["labels"]

        model.train()
        optimizer.zero_grad()
        output = model(*arguments)
        loss_train = self._criterion(output[idx_train], labels[idx_train])
        acc_train = model_meter.accuracy(output[idx_train], labels[idx_train])
        meter.update_vals(loss_train=loss_train.item(), acc_train=acc_train)
        loss_train.backward()
        optimizer.step()

        if not self.conf["fastmode"]:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(*arguments)

        loss_val = self._criterion(output[idx_val], labels[idx_val])
        acc_val = model_meter.accuracy(output[idx_val], labels[idx_val])
        meter.update_vals(loss_val=loss_val.item(), acc_val=acc_val)
        self._logger.debug("%s: Epoch: %03d, %s", model_name, epoch + 1, meter.log_str())

    def _test(self, model_name, model_args, test_idx, meter):
        model, arguments, labels = model_args["model"], model_args["arguments"], model_args["labels"]
        model.eval()
        output = model(*arguments)
        loss_test = functional.nll_loss(output[test_idx], labels[test_idx])
        acc_test = model_meter.accuracy(output[test_idx], labels[test_idx])
        meter.update_diff(output[test_idx], labels[test_idx])
        meter.update_vals(loss_test=loss_test.item(), acc_test=acc_test)
        self._logger.info("%s: Test, %s", model_name, meter.log_str(log_vals=["loss_test", "acc_test"]))

        # self._logger.info("%s Test: loss= %.4f accuracy= %.4f" % (model_name, loss_test.item(), acc_test.item()))
        # return {"loss": loss_test.item(), "acc": acc_test.item()}


def init_seed(seed, cuda=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda is not None:
        torch.cuda.manual_seed(seed)


def aggregate_results(res_list, logger):
    aggregated = {}
    for cur_res in res_list:
        for name, vals in cur_res.items():
            if name not in aggregated:
                aggregated[name] = {}
            for key, val in vals.items():
                if key not in aggregated[name]:
                    aggregated[name][key] = []
                aggregated[name][key].append(val)

    for name, vals in aggregated.items():
        val_list = sorted(vals.items(), key=lambda x: x[0], reverse=True)
        logger.info("*" * 15 + "%s mean: %s", name,
                    ", ".join("%s=%3.4f" % (key, np.mean(val)) for key, val in val_list))
        logger.info("*" * 15 + "%s std: %s", name, ", ".join("%s=%3.4f" % (key, np.std(val)) for key, val in val_list))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='The dataset to use.')
    # parser.add_argument('--prefix', type=str, default="",
    #                     help='The prefix of the products dir name.')

    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not torch.cuda.is_available():
        args.cuda = None
    return args


def main():
    args = parse_args()
    dataset = "cora"  # args.dataset

    seed = random.randint(1, 1000000000)

    conf = {
        "kipf": {"hidden": 16, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4},
        "hidden_layers": [16], "multi_hidden_layers": [100, 20], "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
        "dataset": dataset, "epochs": args.epochs, "cuda": args.cuda, "fastmode": args.fastmode, "seed": seed}

    init_seed(conf['seed'], conf['cuda'])
    dataset_path = os.path.join(PROJ_DIR, "data", dataset)

    products_path = os.path.join(CUR_DIR, "logs", dataset, time.strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("IdansLogger", level=logging.INFO),
        FileLogger("results_%s" % conf["dataset"], path=products_path, level=logging.INFO),
        FileLogger("results_%s_all" % conf["dataset"], path=products_path, level=logging.DEBUG),
    ], name=None)

    data_logger = CSVLogger("results_%s" % conf["dataset"], path=products_path)
    data_logger.set_titles("model_name", "loss", "acc", "train_p", "norm_adj", "feat_type")

    num_iter = 5
    for norm_adj in [True, False]:
        conf["norm_adj"] = norm_adj
        runner = ModelRunner(products_path, dataset_path, conf, logger=logger, data_logger=data_logger)

        for train_p in chain([1], range(5, 90, 10)):
            conf["train_p"] = train_p

            train_p /= 100
            val_p = test_p = (1 - train_p) / 2.
            train_p /= (val_p + train_p)

            runner.loader.split_test(test_p)

            for ft, feat_type in enumerate(["combined", "neighbors", "features"]):
                conf["feat_type"] = feat_type
                results = [runner.run(train_p, feat_type) for _ in range(num_iter)]
                conf_path = os.path.join(runner.products_path,
                                         "t%d_n%d_ft%d.pkl" % (conf["train_p"], norm_adj, ft,))
                pickle.dump({"res": results, "conf": conf}, open(conf_path, "wb"))

    logger.info("Finished")


if __name__ == "__main__":
    main()
