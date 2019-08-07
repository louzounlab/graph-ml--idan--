from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


# from torchnet.meter import AUCMeter
# from torchnet.meter import ConfusionMeter


def accuracy(output, labels, ignore_index=None):
    if ignore_index is not None:
        output = output[labels != ignore_index]
        labels = labels[labels != ignore_index]
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()


# def confusion_matrix(output, labels, num_labels):
#     conf_meter = ConfusionMeter(num_labels)
#     auc_meter = AUCMeter()
#     preds = output.max(1)[1].type_as(labels)
#     conf_meter.add(preds.data.squeeze(), labels.type(torch.LongTensor).data)
#     auc_meter.add(preds.data.squeeze(), labels.data.squeeze())
#     return conf_meter, auc_meter


class ModelMeter:
    def __init__(self, labels, ignore_index=None):
        self._vals = OrderedDict()
        # self._acc = []
        self._labels = labels
        self._ignore_index = ignore_index
        self.clear_diff()

    def update_vals(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self._vals:
                self._vals[key] = []
            self._vals[key].append(val)

    def clear(self):
        self.clear_diff()
        self._vals = OrderedDict()

    def clear_diff(self):
        self._y_preds = np.array([]).reshape(0, len(self._labels)).astype(np.float32)
        self._y_true = np.array([]).reshape(-1).astype(np.float32)
        self._indexes = []

    def update_diff(self, output, targets, change_view=False):
        if change_view:
            targets = targets.view(-1)
            output = output.view(-1)
            output = torch.stack([output, 1 - output], dim=1)

        self._indexes.append(targets.shape[0])
        self._y_preds = np.vstack([self._y_preds, output.cpu().detach().numpy()])
        self._y_true = np.hstack([self._y_true, targets.cpu().detach().numpy()])

    def f1score(self):
        y_pred = np.argmax(self._y_preds, axis=1)
        return metrics.f1_score(self._y_true, y_pred, average='weighted')

    def _auc(self):
        # import pdb; pdb.set_trace()
        # y_pred = np.max(self._y_preds, axis=1)
        y_pred = self._y_preds[:, 1]
        y_true = self._y_true
        if self._ignore_index is not None:
            y_true = y_true[self._y_true != self._ignore_index]
            y_pred = y_pred[self._y_true != self._ignore_index]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)  # , pos_label=1)
        return metrics.auc(fpr, tpr), fpr, tpr

    @property
    def auc(self):
        return self._auc()[0]

    def accuracy(self, output, labels):
        return accuracy(output, labels, self._ignore_index)

    def plot_auc(self, should_show=True):
        roc_auc, fpr, tpr = self._auc()

        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        if should_show:
            plt.show()

        return fig

    def log_str(self, func=None, log_vals=None):
        if func is None:
            vals = OrderedDict((name, val[-1]) for name, val in self._vals.items())
        else:
            vals = OrderedDict((name, func(val)) for name, val in self._vals.items())
        if log_vals is not None:
            vals = OrderedDict((name, vals[name]) for name in log_vals if name in vals)
        return ", ".join(name + ": %.4f" % val for name, val in vals.items())
        # loss: %3.4f, acc: %3.4f", msg, loss, acc)

    def last_val(self, val_name):
        assert val_name in self._vals, "%s not logged" % (val_name,)
        return self._vals[val_name][-1]

    def plot_val(self, val_name, figure_num=None):
        assert val_name in self._vals, "%s not logged" % (val_name,)
        plt.figure(figure_num)
        plt.plot(self._vals[val_name])
        plt.title(val_name.title())
