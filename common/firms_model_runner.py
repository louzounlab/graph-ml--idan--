import os
import time
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim

from loggers import EmptyLogger, multi_logger
from loggers import PrintLogger, FileLogger, CSVLogger
from feature_meta import NODE_FEATURES, DIRECTED_NEIGHBOR_FEATURES
from feature_meta import UNDIRECTED_NEIGHBOR_FEATURES

from common import model_meter
from common.models import GCNCombined, GCN
from common.data_loader import MultiGraphLoader

DATA_PATH = r"/home/benami/git/data/firms/years"
LOGS_PATH = r"/opt/logs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    parser.add_argument('-c', '--common_res_path', default="common_paths.txt",
                        help='Common res path.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose anyway.')
    parser.add_argument('--last', action='store_true', default=False,
                        help='Run test on the given run file.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.cuda = None
    if args.last:
        dir_filter = lambda name: not name.endswith("done")
        args.last = get_recent_logs("part2", dir_filter=dir_filter)
        print("Running on data from: %s" % (args.last,))
    return args


def get_features(feat_type, is_directed, remove_m4=True):
    neighbor_features = UNDIRECTED_NEIGHBOR_FEATURES
    if is_directed:
        neighbor_features = DIRECTED_NEIGHBOR_FEATURES
    all_features = {"neighbors": [neighbor_features],
                    "features": [NODE_FEATURES],
                    "combined": [neighbor_features, NODE_FEATURES],
                    }

    res = dict(y for x in all_features[feat_type] for y in x.items())
    if remove_m4:
        res.pop("motif4", None)
    return res


def sorted_dates(path, fmt, handle=None, reverse=False):
    res = []
    for f in os.listdir(path):
        try:
            new_f = f
            if handle is not None:
                new_f = handle(f)
            dt = datetime.strptime(new_f, fmt)
            res.append((f, dt))
        except Exception:
            pass
    for name, _ in sorted(res, key=lambda x: x[1], reverse=reverse):
        yield os.path.join(path, name)


def get_recent_logs(name, dir_filter=None):
    if dir_filter is None:
        dir_filter = lambda name: name.endswith("done")
    part_dir = os.path.join(LOGS_PATH, name)
    for date_dir in sorted_dates(part_dir, "%Y_%m_%d", reverse=True):
        for run_dir in sorted_dates(date_dir, "%H_%M_%S", handle=lambda x: "_".join(x.split("_")[:3]), reverse=True):
            if dir_filter(run_dir):
                return run_dir
    return None


PART2_RES = "gcn_res"

# def get_last_gcn_res(name, label):
#     prod_path = get_recent_logs(name)
#     if prod_path is not None:
#         return os.path.join(prod_path, label, PART2_RES)
#     return None
#     products_path = os.path.join(raw_prod_path, label)
#     return os.path.join(products_path, "gcn_res")


def get_path_info(name, label):
    raw_prod_path = os.path.join(LOGS_PATH, name, time.strftime("%Y_%m_%d"), time.strftime("%H_%M_%S"))

    products_path = os.path.join(raw_prod_path, label)
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    return {"years": DATA_PATH,
            "label": label,
            "products": products_path,
            "gcn_res": os.path.join(products_path, PART2_RES)}


def finished_path(path):
    base_path = os.path.dirname(path)
    new_name = base_path + "_done"
    os.rename(base_path, new_name)
    return new_name


def get_loggers(name, products_path, is_debug=True, set_titles=True):
    logger = multi_logger([
        PrintLogger("IdansLogger", level=logging.DEBUG if is_debug else logging.INFO),
        FileLogger("results_%s" % name, path=products_path, level=logging.INFO),
        FileLogger("results_%s_all" % name, path=products_path, level=logging.DEBUG),
    ], name=None)

    data_logger = CSVLogger("results_%s" % name, path=products_path)
    if set_titles:
        data_logger.set_titles("feat_type", "year", "loss_val", "loss_test", "acc",
                               "auc_test", "train_p", "norm_adj", "epoch")

    logger.dump_location()
    data_logger.dump_location()
    return logger, data_logger


def init_seed(seed, cuda=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda is not None:
        torch.cuda.manual_seed(seed)


def log_results(data_logger, meter, conf, feat_type, year, products_path, epoch):
    data_logger.log_info(
        year=year,
        loss_val=meter.last_val("loss_val"),
        loss_test=meter.last_val("loss_test"),
        acc=meter.last_val("acc_test"),
        train_p=conf.get("train_p", "unspecified"),
        norm_adj=conf.get("norm_adj", "unspecified"),
        feat_type=feat_type,  # conf.get("feat_type", "unspecified"),
        inp_type=conf.get("inp_type"),
        auc_test=meter.last_val("auc_test"),
        epoch=epoch,
    )

    fig = meter.plot_auc(should_show=False)
    import matplotlib.pyplot as plt
    plot_path = os.path.join(products_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    name = "%s_%s" % (feat_type, year,)
    plt.savefig(os.path.join(plot_path, name))
    plt.clf()
    plt.close(fig)


class ModelRunner:
    def __init__(self, paths, fast_mode, norm_adj, cuda_dev, is_max, logger,
                 data_logger=None, is_debug=False, dtype=torch.double):
        # plt.rcParams.update({'figure.max_open_warning': 0})
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._fast_mode = fast_mode
        self.products_path = paths["products"]
        self._paths = paths
        self._cuda_dev = cuda_dev
        self._dtype = dtype

        self.loaders = MultiGraphLoader(paths, is_max, norm_adj=norm_adj,
                                        logger=self._logger, ignore_index=-1,
                                        cuda_num=cuda_dev, is_debug=is_debug,
                                        dtype=dtype)

        criterion_weight = torch.FloatTensor([1 / 34, 1 / 740])
        if cuda_dev is not None:
            criterion_weight = criterion_weight.cuda(cuda_dev)
        self._criterion_weight = criterion_weight
        self._criterion = torch.nn.NLLLoss(weight=self._criterion_weight, ignore_index=-1)
        # self._criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight, ignore_index=-1)
        self._criterion = self._criterion.type(self._dtype).cuda(self._cuda_dev)
        self._run_label = ""
        self._reset_saved_models()

    def _get_gcn_model(self, mtype, conf, last_layer=False):
        model = {"combined": GCNCombined, "multi": GCN}[mtype]
        conf = conf[mtype]
        extra = {"combined": {"nbow": self.loaders.bow_mx.shape[1]}, "multi": {}}[mtype]
        extra_args = {"combined": ["bow_mx"], "multi": []}[mtype]

        gcn_model = model(nfeat=self.loaders.topo_mx.shape[1],
                          hlayers=conf["multi_hidden_layers"],
                          nclass=self.loaders.num_labels,
                          dropout=conf["dropout"],
                          is_asym=False,
                          last_layer=last_layer,
                          **extra
                          )
        opt = optim.Adam(gcn_model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
        arguments = extra_args + ["topo_mx", "adj_mx"]
        gcn_model.type(self._dtype)
        return gcn_model, opt, arguments

    def _log_results(self, meter, conf, feat_type, year, epoch=None):
        return log_results(self._data_logger, meter, conf, feat_type, year, self.products_path, epoch)

    def _reset_saved_models(self):
        self._best_models = {}

    def _save_best_model(self, name, indices, model_args, epoch=None):
        index = "auc"

        if indices[index] is None:
            return

        if name not in self._best_models or self._best_models[name]["indices"][index] <= indices[index]:
            self._logger.debug("Best model updated: %2.3f (epoch: %s)" % (indices[index], epoch))
            model_state = model_args["model"].state_dict()
            opt_state = model_args["opt"].state_dict()

            self._best_models[name] = {"indices": indices, "model": model_state, "opt": opt_state, "epoch": epoch}

            torch.save(model_state, os.path.join(self.products_path, name + "_model"))
            torch.save(opt_state, os.path.join(self.products_path, name + "_opt"))

    def _load_best_model(self, name, model_args):
        if name in self._best_models:
            epoch = self._best_models[name].get("epoch")
            if epoch is not None:
                index = self._best_models[name]["indices"]
                self._logger.debug("%s: Loaded best model (epoch: %s) (index: %s)" % (name, epoch, index))
            model_args["model"].load_state_dict(self._best_models[name]["model"])
            model_args["opt"].load_state_dict(self._best_models[name]["opt"])

    def _load_model_from_file(self, name, model_args, fpath=None):
        if fpath is None:
            fpath = self.products_path
        model_args["model"].load_state_dict(torch.load(os.path.join(fpath, name + "_model")))
        model_args["opt"].load_state_dict(torch.load(os.path.join(fpath, name + "_opt")))

    def _base_train(self, epoch, model_name, model_args, idx_train, idx_val, meter):
        model, optimizer = model_args["model"], model_args["opt"]
        arguments, labels = model_args["args"], model_args["labels"]
        idx_train = idx_train[labels[idx_train] != -1]
        idx_val = idx_val[labels[idx_val] != -1]

        model.train()
        optimizer.zero_grad()
        output = model(*arguments)

        loss_train = self._criterion(output[idx_train], labels[idx_train])
        acc_train = model_meter.accuracy(output[idx_train], labels[idx_train])
        meter.update_vals(loss_train=loss_train.item(), acc_train=acc_train)
        loss_train.backward()
        optimizer.step()

        if not self._fast_mode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(*arguments)

        loss_val = auc_val = acc_val = None
        if idx_val.cpu().numpy().any():
            loss_val = self._criterion(output[idx_val], labels[idx_val]).item()
            acc_val = model_meter.accuracy(output[idx_val], labels[idx_val])
            meter.update_diff(output[idx_val], labels[idx_val])
            auc_val = meter.auc
            meter.update_vals(loss_val=loss_val, acc_val=acc_val, auc_val=auc_val)

        self._logger.debug("%s: Epoch: %03d, %s", model_name, epoch + 1, meter.log_str())
        return {"loss": loss_val, "acc": acc_val, "auc": auc_val}

    def _base_test(self, model_name, model_args, idx_test, meter, idx_val=None):
        model, arguments, labels = model_args["model"], model_args["args"], model_args["labels"]
        idx_test = idx_test[labels[idx_test] != -1]

        model.eval()
        output = model(*arguments)
        loss_test = self._criterion(output[idx_test], labels[idx_test]).item()
        acc_test = model_meter.accuracy(output[idx_test], labels[idx_test])

        loss_val = None
        if idx_val.cpu().numpy().any():
            loss_val = self._criterion(output[idx_val], labels[idx_val]).item()

        meter.update_diff(output[idx_test], labels[idx_test])
        meter.update_vals(loss_test=loss_test, acc_test=acc_test, auc_test=meter.auc,
                          loss_val=loss_val)
        self._logger.info("%s: Test, %s", model_name, meter.log_str(log_vals=["loss_val", "loss_test", "acc_test", "auc_test"]))


#        self._data_logger.log_info(
#            name=name,
#            loss=meter.last_val("loss_test"),
#            acc=meter.last_val("acc_test"),
#            train_p=conf["train_p"],
#            norm_adj=conf["norm_adj"],
#            feat_type=conf["feat_type"],
#            auc_test=meter.last_val("auc_test"),
#            epoch=epoch,
#        )
#        fig = meter.plot_auc(should_show=False)
#        import matplotlib.pyplot as plt
#        plot_path = os.path.join(self.products_path, "plots")
#        if not os.path.exists(plot_path):
#            os.makedirs(plot_path)
#        plt.savefig(os.path.join(plot_path, name))
#        plt.clf()
#        plt.close(fig)
