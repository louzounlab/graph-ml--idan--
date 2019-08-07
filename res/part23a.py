import os
import random
import numpy as np

import torch
from torch.autograd import Variable

from common import model_meter
from common.firms_conf import general_conf
from common.firms_model_runner import get_features, get_loggers
from common.firms_model_runner import init_seed, ModelRunner, parse_args
from common.firms_model_runner import get_path_info, finished_path


IS_DEBUG = False
# =============================================================================
# ================================ Phase 2 & 3a ===============================
# =============================================================================


class PhaseTwoThreeRunner(ModelRunner):
    def _evaluate_model(self, model, arguments, model_name):
        model.last_layer = False
        model.eval()
        out_path = self._paths["gcn_res"] + "_" + model_name

        outputs = []
        with torch.no_grad():
            for loader in self.loaders:
                args = [Variable(getattr(loader, arg)).cuda(self._cuda_dev) for arg in arguments]
                res = model(*args)
                outputs.append(res.cpu().detach().numpy())
                del args
                torch.cuda.empty_cache()
        outputs = np.stack(outputs, axis=0)
        np.save(out_path, outputs)
        self._logger.info("Dumped to %s" % (out_path,))
        model.last_layer = True

    def run(self, conf, run_label):
        features_meta = get_features(conf["feat_type"], is_directed=self.loaders.is_graph_directed)
        self.loaders.split_train(conf["train_p"], features_meta)

        models = {name: self._get_gcn_model(name, conf, last_layer=True)
                  for name in ["combined", "multi"]}

        models = {name: {"model": args[0], "opt": args[1], "args": args[2]}
                  for name, args in models.items()}

        for meta in models.values():
            meta["model"].cuda(self._cuda_dev)

        self._reset_saved_models()
        meters = {name: model_meter.ModelMeter(self.loaders.distinct_labels) for name in models}

        train_idx, val_idx = self.loaders.train_idx(), self.loaders.val_idx()
        if conf["last_run"]:
            fpath = conf["last_run"]
            for name, model_meta in models.items():
                self._load_model_from_file(name, model_meta, fpath)
        else:
            # Train model
            for epoch in range(conf["epochs"]):
                for name, model_meta in models.items():
                    results = []
                    for loader in self.loaders:
                        results.append(self._train(epoch, loader, name, model_meta, train_idx, val_idx, meters[name]))

                    indices = {index: [res[index] for res in results] for index in results[0]}
                    indices = {index: 0 if None in vals else np.mean(vals) for index, vals in indices.items()}
                    self._save_best_model(name, indices, model_meta, epoch=epoch)

        for name, model_meta in models.items():
            if not conf["last_run"]:
                self._load_best_model(name, model_meta)
            self._evaluate_model(model_meta["model"], model_meta["args"], name)

        # Testing
        test_idx = self.loaders.test_idx()
        for name, model_meta in models.items():
            meter = meters[name]
            for loader in self.loaders:
                cur_name = "%s_%s" % (loader.name, name,)
                self._test(loader, cur_name, model_meta, test_idx, meter, val_idx)
                self._log_results(meter, conf, name, loader.name)

        self._logger.dump_location()
        self._data_logger.dump_location()

        return meters

    def _train(self, epoch, loader, model_name, model_meta, idx_train, idx_val, meter):
        model_name = "{} {}".format(model_name, loader.name)

        model_args = {
            "model": model_meta["model"], "opt": model_meta["opt"],
            "args": [Variable(getattr(loader, arg)).cuda(self._cuda_dev) for arg in model_meta["args"]],
            "labels": Variable(loader.labels).cuda(self._cuda_dev)
        }
        res = self._base_train(epoch, model_name, model_args, idx_train, idx_val, meter)
        torch.cuda.empty_cache()
        return res

    def _test(self, loader, model_name, model_meta, test_idx, meter, idx_val):
        model_args = {"model": model_meta["model"],
                      "labels": Variable(loader.labels).cuda(self._cuda_dev),
                      "args": [Variable(getattr(loader, arg)).cuda(self._cuda_dev) for arg in model_meta["args"]],
                      }
        res = self._base_test(model_name, model_args, test_idx, meter, idx_val=idx_val)
        torch.cuda.empty_cache()
        return res


def main(args, paths, label, logger, data_logger):
    seed = random.randint(1, 1000000000)
#    conf = {
#        "kipf": {"hidden": 16, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4},
#        "hidden_layers": [16], "multi_hidden_layers": [100, 35], "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
#        "norm_adj": True, "feat_type": "combined",
#        "dataset": "firms", "epochs": 200, "cuda": args.cuda, "fastmode": args.fastmode, "seed": seed}
    conf = general_conf
    conf.update({"seed": seed, "cuda": args.cuda, "norm_adj": True,
                 "dataset": "firms", "feat_type": "combined",
                 "last_run": os.path.join(args.last, label) if args.last else args.last})
    if IS_DEBUG:
        conf["epochs"] = 2

    init_seed(conf['seed'], conf['cuda'])

    # index = 0
    # num_iter = 1

    runner = PhaseTwoThreeRunner(paths, args.fastmode, conf["norm_adj"], conf["cuda"], conf["is_max"],
                                 logger=logger, data_logger=data_logger, is_debug=IS_DEBUG)
    runner.loaders.split_test(conf["test_p"])
    runner.run(conf, '0')

    # for i in range(num_iter):
    #     runner.run(conf, str(index + i))
    # results = [runner.run(conf, str(index + i)) for i in range(num_iter)]
    # index += num_iter
    # conf_path = os.path.join(runner.products_path, "t%d_n%d_ft%d.pkl" % (conf["train_p"], norm_adj, ft,))
    # pickle.dump({"res": results, "conf": conf}, open(conf_path, "wb"))


if __name__ == "__main__":
    inp_args = parse_args()
    path_info = get_path_info("part2", "top")
    logger, data_logger = get_loggers("firms", path_info["products"], is_debug=IS_DEBUG or inp_args.verbose)
    open(inp_args.common_res_path, "a").write("Started: %s\n" % (data_logger.get_location()))
    main(inp_args, path_info, "top", logger, data_logger)

    logger.info("Finished")
    if not IS_DEBUG:
        prod_path = finished_path(path_info["products"])
        open(inp_args.common_res_path, "a").write("Finished: %s\n" % (prod_path,))


# def aggregate_results(res_list, logger):
#    aggregated = {}
#    for cur_res in res_list:
#        for name, vals in cur_res.items():
#            if name not in aggregated:
#                aggregated[name] = {}
#            for key, val in vals.items():
#                if key not in aggregated[name]:
#                    aggregated[name][key] = []
#                aggregated[name][key].append(val)
#
#    for name, vals in aggregated.items():
#        val_list = sorted(vals.items(), key=lambda x: x[0], reverse=True)
#        logger.info("*" * 15 + "%s mean: %s", name,
#                    ", ".join("%s=%3.4f" % (key, np.mean(val)) for key, val in val_list))
#        logger.info("*" * 15 + "%s std: %s", name, ", ".join("%s=%3.4f" % (key, np.std(val)) for key, val in val_list))
