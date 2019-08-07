import os
import pickle
import random

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from common import model_meter
from common.models_rnn import GCNRNN
from common.firms_conf import general_conf
from common.firms_model_runner import ModelRunner, get_features, get_loggers, init_seed, parse_args
from common.firms_model_runner import get_path_info, finished_path


DTYPE = torch.float32
IS_DEBUG = False
# =======================================================================================
# ======================================= Phase 4 =======================================
# =======================================================================================


class RNNGCNRunner(ModelRunner):
    # def __init__(self, *args, **kwargs):
        # super(RNNGCNRunner, self).__init__(*args, **kwargs)
        # self._criterion = torch.nn.CrossEntropyLoss(weight=self._criterion_weight, ignore_index=-1)

    def _get_rnn_model(self, conf, mtype):
        gcn_model, _, arguments = self._get_gcn_model(mtype, conf, last_layer=False)
        rnn_model = GCNRNN(gcn_model, layer_size=51, num_layers=2,
                           cuda_num=conf.get("cuda"), dtype=self._dtype,
                           nclasses=self.loaders.num_labels)
        opt = optim.LBFGS(rnn_model.parameters(), lr=0.005)
        return rnn_model, opt, arguments

    def run(self, train_p, conf, run_label, mtype):
        # self._criterion.type(DTYPE)
        self._run_label = str(run_label)
        features_meta = get_features(conf["feat_type"], is_directed=self.loaders.is_graph_directed)
        self.loaders.split_train(train_p, features_meta)

        rnn_model, opt, arguments = self._get_rnn_model(conf, mtype)
        rnn_model.type(DTYPE)

        if self._cuda_dev is not None:
            rnn_model.cuda(self._cuda_dev)

        model_meta = {"model": rnn_model, "opt": opt}
        model_name = "rnn_%s" % (conf["feat_type"], )

        self._res_path = os.path.join(self.products_path, "res")
        if not os.path.exists(self._res_path):
            os.makedirs(self._res_path)

        # Train model
        meter = model_meter.ModelMeter(self.loaders.distinct_labels, ignore_index=-1)
        for epoch in range(conf["epochs"]):
            indices = self._train(rnn_model, opt, arguments, meter, epoch)
            self._save_best_model(model_name, indices, model_meta, epoch=epoch)

            if 0 == (epoch + 1) % 5:
                with torch.no_grad():
                    test_meter = model_meter.ModelMeter(self.loaders.distinct_labels, ignore_index=-1)
                    self._test(rnn_model, arguments, conf, meter=test_meter, epoch=epoch)

        self._load_best_model(model_name, model_meta)
        with torch.no_grad():
            test_meter = model_meter.ModelMeter(self.loaders.distinct_labels, ignore_index=-1)
            self._test(rnn_model, arguments, conf, meter=test_meter)

    def _train(self, model, optimizer, arguments, meter, epoch):
        train_idx, val_idx = self.loaders.train_idx(), self.loaders.val_idx()
        labels = Variable(self.loaders.labels).cuda(self._cuda_dev)
        train_target = labels[train_idx].view(-1)
        val_target = labels[val_idx].view(-1)

        model.train()
        indices = {"loss": None, "acc": None, "auc": None}

        self._cur_run = 0

        def closure():
            optimizer.zero_grad()

            for i, loader in enumerate(self.loaders):
                # print("%s-%d" % (loader.name, i+1))
                # torch.cuda.empty_cache()
                args = [Variable(getattr(loader, arg)).type(DTYPE).cuda(self._cuda_dev) for arg in arguments]
                is_first = (i == 0)
                is_last = (i == (len(self.loaders) - 1))
                output = model(*args, init=is_first, clear=is_last, export=is_last)

            train_pred = output[train_idx].view(-1, 2)
            # train_pred = torch.stack([train_pred, 1 - train_pred], dim=1)

            if False:
                np.save(os.path.join(self._res_path, "%d_%d" % (epoch, self._cur_run,)),
                        train_pred.cpu().detach().numpy()
                        )
                self._cur_run += 1

            loss_train = self._criterion(train_pred, train_target)

            acc_train = meter.accuracy(train_pred, train_target)
            meter.update_vals(loss_train=loss_train.item(), acc_train=acc_train)
            loss_train.backward()

            if val_idx.cpu().numpy().any():
                meter.clear_diff()
                val_pred = output[val_idx].view(-1, 2)
                # val_pred = torch.stack([val_pred, 1 - val_pred], dim=1)

                indices["loss"] = self._criterion(val_pred, val_target).item()
                indices["acc"] = meter.accuracy(val_pred, val_target)
                meter.update_diff(val_pred, val_target)
                indices["auc"] = meter.auc
                meter.update_vals(loss_val=indices["loss"], acc_val=indices["acc"], auc_val=indices["auc"])

            torch.cuda.empty_cache()

            # self._logger.debug("%s: Epoch: %03d, %s", model_name, epoch + 1, meter.log_str())
            self._logger.debug("%d. Train: %s", epoch, meter.log_str())
            return loss_train

        optimizer.step(closure)
        return indices

    def _test2(self, model, arguments, conf, meter, epoch=None):
        test_idx = self.loaders.test_idx()
        labels = Variable(self.loaders.labels).cuda(self._cuda_dev)
        test_target = labels[test_idx].view(-1)
        model.eval()

        for i, loader in enumerate(self.loaders):
            args = [Variable(getattr(loader, arg)).type(DTYPE).cuda(self._cuda_dev) for arg in arguments]
            output = model(*args,
                           init=(0 == i),
                           clear=(i == (len(self.loaders) - 1)),
                           export=(i == (len(self.loaders) - 1)))

        meter.clear()
        pred = output[test_idx].view(-1, 2)
        # pred = torch.stack([pred, 1 - pred], dim=1)

        loss_test = self._criterion(pred, test_target)
        acc_test = meter.accuracy(pred, test_target)
        meter.update_diff(pred, test_target, change_view=False)
        meter.update_vals(loss_test=loss_test.item(), acc_test=acc_test, auc_test=meter.auc)
        self._logger.info("Test %d: %s", i + 1, meter.log_str(log_vals=["loss_test", "acc_test", "auc_test"]))
        self._log_results(meter, conf, loader.name, epoch=epoch)
        torch.cuda.empty_cache()

    def _test(self, model, arguments, conf, meter, epoch=None):
        val_idx, test_idx = self.loaders.val_idx(), self.loaders.test_idx()
        labels = Variable(self.loaders.labels).cuda(self._cuda_dev)
        val_target = labels[val_idx]
        test_target = labels[test_idx]
        model.eval()

        for i, loader in enumerate(self.loaders):
            args = [Variable(getattr(loader, arg)).type(DTYPE).cuda(self._cuda_dev) for arg in arguments]
            output = model(*args, init=(0 == i), clear=True, export=True)

            meter.clear()
            tpred = output[test_idx].view(-1, 2)
            ttarget = test_target[:, i]

            loss_test = self._criterion(tpred, ttarget)
            acc_test = meter.accuracy(tpred, ttarget)

            loss_val = None
            if val_idx.cpu().numpy().any():
                vpred = output[val_idx].view(-1, 2)
                vtarget = val_target[:, i]
                loss_val = self._criterion(vpred, vtarget).item()

            meter.update_diff(tpred, ttarget)
            meter.update_vals(loss_test=loss_test.item(), acc_test=acc_test, auc_test=meter.auc,
                              loss_val=loss_val)
            self._logger.info("Test %d: %s", i + 1, meter.log_str(log_vals=["loss_test", "acc_test", "auc_test"]))
            self._log_results(meter, conf, conf["feat_type"], loader.name, epoch=epoch)
            torch.cuda.empty_cache()


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


def main(args, paths, logger, data_logger):
    dataset = "firms"

    seed = random.randint(1, 1000000000)

    conf = general_conf
    conf.update({"cuda": args.cuda, "dataset": dataset, "seed": seed, "norm_adj": True})
    if IS_DEBUG:
        conf["epochs"] = 2

    init_seed(conf['seed'], conf['cuda'])

    #######################
    # "gcn_res": os.path.realpath(os.path.join(base_paths["data"], "..", "gcn_res"))}

    # products_path = path_info["products"]
    # logger = multi_logger([
    #     PrintLogger("IdansLogger", level=logging.DEBUG),
    #     FileLogger("results_%s" % conf["dataset"], path=products_path, level=logging.INFO),
    #     FileLogger("results_%s_all" % conf["dataset"], path=products_path, level=logging.DEBUG),
    # ], name=None)

    # data_logger = CSVLogger("results_%s" % conf["dataset"], path=products_path)
    # data_logger.set_titles("name", "epoch", "loss", "acc", "auc_test", "train_p", "norm_adj", "feat_type")

    index = 0
    num_iter = 1
    runner = RNNGCNRunner(paths, args.fastmode, conf["norm_adj"], conf["cuda"], conf["is_max"],
                          logger=logger, data_logger=data_logger, is_debug=IS_DEBUG, dtype=DTYPE)

    runner.loaders.split_test(1 - (conf["train_p"] / 100))

    results = [runner.run(1, conf, str(index + i), "combined") for i in range(num_iter)]
    index += num_iter
    conf_path = os.path.join(runner.products_path, "t%d_ft.pkl" % (conf["train_p"],))
    pickle.dump({"res": results, "conf": conf}, open(conf_path, "wb"))


if __name__ == "__main__":
    inp_args = parse_args()
    path_info = get_path_info("part4", "top")
    logger, data_logger = get_loggers("firms", path_info["products"], is_debug=IS_DEBUG or inp_args.verbose)
    open(inp_args.common_res_path, "a").write("Started: %s\n" % (data_logger.get_location()))
    main(inp_args, path_info, logger, data_logger)

    logger.info("Finished")
    if not IS_DEBUG:
        prod_path = finished_path(path_info["products"])
        open(inp_args.common_res_path, "a").write("Finished: %s\n" % (prod_path,))
    # main(inp_args, PATHS, "appear")

# def _evaluate_model(self, model, arguments):
#     model.last_layer = False
#     model.eval()
#
#     outputs = []
#     for loader in self.loaders:
#         args = [Variable(getattr(loader, arg)) for arg in arguments]
#         outputs.append(model(*args))
#     outputs = np.stack([x.cpu().detach().numpy() for x in outputs], axis=0)
#     np.save(self._paths["gcn_res"] + self._run_label, outputs)
#     model.last_layer = True
#     self._logger.info("Dumped")

# train_p /= 100
# val_p = test_p = (1 - train_p) / 2.
# train_p /= (val_p + train_p)
# test_p = 1 - train_p
