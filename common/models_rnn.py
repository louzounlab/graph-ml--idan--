import torch
import torch.nn as nn
import torch.nn.functional as functional


class RNN(nn.Module):
    def __init__(self, n_features, layer_size, num_layers, nclasses=1, cuda_num=None, dtype=torch.double):
        super(RNN, self).__init__()
        self._layer_size = layer_size
        self._n_layers = num_layers
        self._n_features = n_features
        self._cuda_num = cuda_num
        self._dtype = dtype
        self._nclasses = nclasses

        hiddens = [n_features] + [layer_size] * num_layers
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(i, o) for i, o in zip(hiddens[:-1], hiddens[1:])])
        self.linear = nn.Linear(layer_size, self._nclasses)
        # self.sigmoid = nn.Sigmoid()

    def _init_initials(self, batch_size):
        return {i: {"h_t": torch.zeros((batch_size, self._layer_size), dtype=self._dtype).cuda(self._cuda_num),
                    "c_t": torch.zeros((batch_size, self._layer_size), dtype=self._dtype).cuda(self._cuda_num)}
                for i in range(self._n_layers)}

    def forward_model(self, h_t, initials):
        for i, layer in enumerate(self.lstm_layers):
            h_t, c_t = layer(h_t, (initials[i]["h_t"], initials[i]["c_t"]))
            initials[i] = {"h_t": h_t, "c_t": c_t}
        # return self.sigmoid(h_t)
        # return self.sigmoid(self.linear(h_t))
        x = self.linear(h_t)
        # import pdb; pdb.set_trace()
        res = functional.log_softmax(x, dim=1)
        # import pdb; pdb.set_trace()
        return res

    def forward(self, x, future=0):
        outputs = []

        initials = self._init_initials(x.size(0))

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            output = self.forward_model(input_t.squeeze(dim=1), initials)
            outputs.append(output)
            # plt.figure()
            # plt.plot(input_t.cpu().detach().numpy().transpose())
            # plt.show()

        for i in range(future):  # if we should predict the future
            output = self.forward_model(output, initials)
            outputs.append(output)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class GCNRNN(RNN):
    def __init__(self, gcn_model, layer_size, num_layers, cuda_num=None, **kwargs):
        super(GCNRNN, self).__init__(gcn_model.n_output, layer_size, num_layers, cuda_num=cuda_num, **kwargs)
        self._gcn_model = gcn_model
        self._outputs = []
        self._initials = {}

    def forward(self, *gcn_params, init=False, export=False, clear=False):
        gcn_out = self._gcn_model(*gcn_params)

        if init:
            self._outputs = []
            self._initials = self._init_initials(gcn_out.size(0))
        self._outputs.append(self.forward_model(gcn_out.squeeze(dim=1), self._initials))

        if export:
            res = torch.stack(self._outputs, 1).squeeze(2)

        if clear:
            self._outputs = []

        if export:
            return res
