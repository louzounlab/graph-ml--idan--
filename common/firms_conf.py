kipf_conf = {
    "hidden": 16, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4
}

general_conf = {
    "combined": {
        "hidden_layers": [16], "multi_hidden_layers": [100, 35],
        "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
    },
    "multi": {
        "hidden_layers": [16], "multi_hidden_layers": [100, 35],
        "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
    },
    "epochs": 200, "is_max": False, "feat_type": "combined",
    "train_p": 60, "test_p": 20, "val_p": 20
}

# {
#     "hidden_layers": [16], "multi_hidden_layers": [100, 35], "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
#     "dataset": dataset, "epochs": args.epochs, "cuda": args.cuda, "fastmode": args.fastmode, "seed": seed}

# conf = {
#     "hidden_layers": [16], "multi_hidden_layers": [100, 20], "dropout": 0.6, "lr": 0.01, "weight_decay": 0.001,
#     "dataset": data_set, "epochs": args.epochs, "cuda": args.cuda,
#     "seed": seed, "norm_adj": True, "feat_type": "combined",
# }

# conf = {
#     "kipf": {},
#     "hidden_layers": [16],
#     "dataset": data_set,
#     "epochs": args.epochs, "cuda": args.cuda, "fastmode": args.fastmode, "seed": seed
# }
