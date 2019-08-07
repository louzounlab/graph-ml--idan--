import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import model_meter
from common.models_rnn import RNN
from common.firms_model_runner import log_results, get_loggers, get_recent_logs, init_seed, parse_args
from common.firms_model_runner import get_path_info, finished_path, PART2_RES


IS_DEBUG = False
# =============================================================================
# ================================== Phase 3b =================================
# =============================================================================


class RNNDataLoader:
    def __init__(self):
        self._data = np.array([])
        self._n_train = 0

    @property
    def train_input(self):
        return torch.from_numpy(self._data[:self._n_train, :-1])

    @property
    def train_target(self):
        raise NotImplementedError()

    @property
    def test_input(self):
        return torch.from_numpy(self._data[self._n_train:, :-1])

    @property
    def test_target(self):
        return 1

    @property
    def n_samples(self):
        return self._data.shape[0]

    @property
    def n_entries(self):
        return self._data.shape[1]

    @property
    def n_features(self):
        return self._data.shape[2]

    def split_train_test(self, ratio):
        self._n_train = int(ratio * self.n_samples)
        #
        # input = input.cuda()
        # target = target.double().cuda()
        # test_input = test_input.cuda()
        # test_target = test_target.cuda()


class RNNGCNLoader(RNNDataLoader):
    def __init__(self, data_path, index_path, labels_path):
        super(RNNGCNLoader, self).__init__()
        self._data = np.load(data_path).transpose(1, 0, 2).astype(np.float64)  # n x n_seq x n_feat
        self._labels = np.load(labels_path)  # n x 17
        self._labels = self._labels[:, :self._data.shape[1]]
        indexes = pickle.load(open(index_path, "rb"))
        self._train_idx = indexes["base"]
        self._val_idx = indexes["val"]
        self._test_idx = indexes["test"]

    @property
    def train_input(self):
        return torch.from_numpy(self._data[self._train_idx]).double()

    @property
    def train_target(self):
        return torch.from_numpy(self._labels[self._train_idx]).long()

    @property
    def val_input(self):
        return torch.from_numpy(self._data[self._val_idx]).double()

    @property
    def val_target(self):
        return torch.from_numpy(self._labels[self._val_idx]).long()

    @property
    def test_input(self):
        return torch.from_numpy(self._data[self._test_idx]).double()

    @property
    def test_target(self):
        return torch.from_numpy(self._labels[self._test_idx]).long()


class MyLoss(nn.CrossEntropyLoss):
    def forward(self, pred, target):
        if pred.shape == target.shape:
            target = target.view(-1)
            pred = pred.view(-1)
            pred = torch.stack([pred, 1 - pred], dim=1)
            # from collections import Counter; print(Counter(target.cpu().detach().numpy()))
        return super(MyLoss, self).forward(pred, target)


def rnn_main(loader, conf, logger, data_logger, products_path):
    # set random seed to 0
    init_seed(conf["seed"], cuda=conf["cuda"])

    # load data and make training set
    n_feat = loader.n_features

    loader.split_train_test(conf["train_p"])
    train_input = loader.train_input
    train_target = loader.train_target.view(-1)
    val_input = loader.val_input
    val_target = loader.val_target
    test_input = loader.test_input
    test_target = loader.test_target

    # build the model
    seq = RNN(n_features=n_feat, layer_size=51, num_layers=2,
              nclasses=2, cuda_num=conf["cuda"])
    seq.double()

    if True:
        # criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss(ignore_index=-1)
        criterion_weight = torch.DoubleTensor([1 / 34, 1 / 740])
        if conf["cuda"] is not None:
            criterion_weight = criterion_weight.cuda(conf["cuda"])

        criterion = MyLoss(weight=criterion_weight, ignore_index=-1)
        optimizer = optim.LBFGS(seq.parameters(), lr=0.1)
        # optimizer = optim.Adam(seq.parameters(), lr=0.2, weight_decay=0.0001)
    else:
        # use LBFGS as optimizer since we can load the whole data to train
        criterion = nn.MSELoss()
        optimizer = optim.LBFGS(seq.parameters(), lr=0.3)
    # begin to train

    # Moving to cuda
    if conf["cuda"] is not None:
        train_input = train_input.cuda(conf["cuda"])
        train_target = train_target.cuda(conf["cuda"])
        val_input = val_input.cuda(conf["cuda"])
        val_target = val_target.cuda(conf["cuda"])
        train_val_target = val_target.cuda(conf["cuda"]).view(-1)
        test_input = test_input.cuda(conf["cuda"])
        test_target = test_target.cuda(conf["cuda"])
        seq.cuda(conf["cuda"])

    for epoch in range(conf["epochs"]):
        logger.info('STEP: %d' % (epoch + 1,))
        meter = model_meter.ModelMeter([0, 1], ignore_index=-1)

        indices = {"loss": None, "acc": None, "auc": None}

        # for x in range(50):
        def closure():
            optimizer.zero_grad()
            train_pred = seq(train_input)

            train_pred = train_pred.view(-1, 2)
            # train_pred = torch.stack([train_pred, 1 - train_pred], dim=1)

            loss_train = criterion(train_pred, train_target)

            acc_train = meter.accuracy(train_pred, train_target)
            meter.update_vals(loss_train=loss_train.item(), acc_train=acc_train)
            loss_train.backward()

            if val_input.cpu().numpy().any():
                val_pred = seq(val_input)
                val_pred = val_pred.view(-1, 2)
                # meter.clear_diff()
                # val_pred = torch.stack([val_pred, 1 - val_pred], dim=1)

                indices["loss"] = criterion(val_pred, train_val_target).item()
                indices["acc"] = meter.accuracy(val_pred, train_val_target)
                meter.update_diff(val_pred, train_val_target)
                indices["auc"] = meter.auc
                meter.update_vals(loss_val=indices["loss"], acc_val=indices["acc"], auc_val=indices["auc"])

            logger.debug("%d. Train: %s", epoch, meter.log_str())
            return loss_train

            # optimizer.step()

        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            test_pred = seq(test_input)

            val_pred = None
            if val_input.cpu().numpy().any():
                val_pred = seq(val_input)

            exec_test(criterion,
                      test_pred, test_target,
                      val_pred, val_target,
                      logger, data_logger, conf, products_path, meter=meter, epoch=epoch)


def exec_test(criterion,
              test_pred, test_target,
              val_pred, val_target,
              logger, data_logger, conf, products_path, meter=None, epoch=None):
    # val_pred = val_pred.view(-1, 2)
    # import pdb; pdb.set_trace()
    for i in range(test_pred.shape[1]):
        year = 1996 + i
        meter = model_meter.ModelMeter([0, 1], ignore_index=-1)
        tpred = test_pred[:, i]
        ttarget = test_target[:, i]
        loss_test = criterion(tpred, ttarget)

        # logger.info('test loss: %f' % loss_test.item())
        acc_test = meter.accuracy(tpred, ttarget)

        loss_val = None
        if val_pred is not None:
            vpred = val_pred[:, i]
            vtarget = val_target[:, i]
            loss_val = criterion(vpred, vtarget).item()

        meter.update_diff(tpred, ttarget)
        meter.update_vals(loss_test=loss_test.item(), acc_test=acc_test, auc_test=meter.auc,
                          loss_val=loss_val)

        name = "%d_%s" % (year, conf["mtype"],)
        logger.info("Test %s: %s", name, meter.log_str(log_vals=["loss_val", "loss_test", "acc_test", "auc_test"]))

        # meter.plot_auc(should_show=True)
        # log_results(data_logger, meter, conf, name, products_path, None)
        log_results(data_logger, meter, conf, conf["mtype"], str(year), products_path, epoch)


# def exec_test1(criterion, test_pred, test_target, logger, meter=None):
def exec_test1(criterion, test_pred, test_target, logger, data_logger, conf, products_path, meter=None, epoch=None):
    test_pred = test_pred.view(-1)
    test_pred = torch.stack([test_pred, 1 - test_pred], dim=1)

    test_target = test_target.view(-1)
    loss_test = criterion(test_pred, test_target)
    # logger.info('test loss: %f' % loss_test.item())
    acc_test = meter.accuracy(test_pred, test_target)
    meter.update_vals(loss_test=loss_test.item(), acc_test=acc_test)

    meter.update_diff(test_pred, test_target)
    logger.info("Test: %s", meter.log_str(log_vals=["loss_test", "acc_test"]))

    meter.plot_auc(should_show=True)


def main(args, paths, label, logger, data_logger):
    seed = random.randint(1, 1000000000)

    recent_log_dir = get_recent_logs("part2")
    assert recent_log_dir is not None, "Couldn't find recent part2 log dir"
    gcn_res = os.path.join(recent_log_dir, label, PART2_RES)

    base_path = os.path.dirname(paths["years"])
    conf = {"seed": seed, "train_p": 0.7, "cuda": args.cuda, "mtype": None, "epochs": 20}
    if IS_DEBUG:
        conf["epochs"] = 2

    for mtype in ["combined", "multi"]:
        conf["mtype"] = mtype
        data_file = "%s_%s.npy" % (gcn_res, conf["mtype"],)
        logger.info("Using data file: %s" % (data_file, ))

        loader = RNNGCNLoader(
                data_file,
                os.path.join(base_path, "split.pkl"),
                os.path.join(base_path, "labels.npy")
                )

        rnn_main(loader, conf, logger, data_logger, paths["products"])


if __name__ == '__main__':
    cur_label = "top"
    inp_args = parse_args()
    path_info = get_path_info("part3", cur_label)
    logger, data_logger = get_loggers("firms", path_info["products"], is_debug=IS_DEBUG or inp_args.verbose)
    open(inp_args.common_res_path, "a").write("Started: %s\n" % (data_logger.get_location()))
    main(inp_args, path_info, cur_label, logger, data_logger)

    logger.info("Finished")
    if not IS_DEBUG:
        prod_path = finished_path(path_info["products"])
        open(inp_args.common_res_path, "a").write("Finished: %s\n" % (prod_path,))
