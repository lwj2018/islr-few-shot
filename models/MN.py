import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.metricUtils import euclidean_metric
from utils.fewshotUtils import create_nshot_task_label_t

class MN(nn.Module):
    def __init__(self,baseModel,
                lstm_layers=1,
                lstm_input_size=1600,
                unrolling_steps=2,
                train_way=20,test_way=5,
                shot=5,query=5,query_val=15,
                ):
        super(MN,self).__init__()
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.query = query
        self.query_val = query_val

        self.baseModel = baseModel
        self.g = BidrectionalLSTM(lstm_input_size,lstm_layers)
        self.f = AttentionLSTM(lstm_input_size,unrolling_steps)

    def forward(self, support, queries, mode='train'):
        if mode=='train':
            way = self.train_way
            query = self.query
        else:
            way = self.test_way
            query = self.query_val
        # Concatenate
        support = support.reshape(self.shot,way,-1)
        support = support.permute(1,0,2).reshape(way * self.shot,-1)
        queries = queries.reshape(query,way,-1)
        queries = queries.permute(1,0,2).reshape(way * query,-1)
        x = torch.cat([support,queries],0)
        # Embed all samples
        embeddings = self.baseModel(x)

        # Samples are ordered by the NShotWrapper class as follows:
        # k lots of n support samples from a particular class
        # k lots of q query samples from those classes
        support = embeddings[:self.shot * way]
        queries = embeddings[self.shot * way:]

        # LSTM requires input of shape (seq_len, batch, input_size). `support` is of
        # shape (k_way * n_shot, embedding_dim) and we want the LSTM to treat the
        # support set as a sequence so add a single dimension to transform support set
        # to the shape (k_way * n_shot, 1, embedding_dim) and then remove the batch dimension
        # afterwards

        # Calculate the fully conditional embedding, g, for support set samples as described
        # in appendix A.2 of the paper. g takes the form of a bidirectional LSTM with a
        # skip connection from inputs to outputs
        support, _, _ = self.g(support.unsqueeze(1))
        support = support.squeeze(1)

        # Calculate the fully conditional embedding, f, for the query set samples as described
        # in appendix A.1 of the paper.
        queries = self.f(support, queries)

        # Efficiently calculate distance between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way * n_shot) = (num_queries, num_support)
        distances = euclidean_metric(queries, support)

        # Calculate "attention" as softmax over support-query distances
        attention = (distances)#.softmax(dim=1)

        # Calculate predictions as in equation (1) from Matching Networks
        # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
        y_pred = matching_net_predictions(attention, self.shot, way, query)
        # y_pred = y_pred.log()
        label = create_nshot_task_label_t(way,query).cuda()
        return y_pred, label

    def get_optim_policies(self, lr):
        return [
            {'params':self.parameters(),'lr':lr},
        ]

def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.

    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.

    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
        n: Number of support set samples per class, n-shot
        k: Number of classes in the episode, k-way
        q: Number of query samples per-class

    # Returns
        y_pred: Predicted class probabilities
    """
    if attention.shape != (q * k, k * n):
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, k)

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    y = create_nshot_task_label_t(k, n).unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1)

    y_pred = torch.mm(attention, y_onehot.cuda().float())

    return y_pred
        

class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.
        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.
        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().float()
        c = torch.zeros(batch_size, embedding_dim).cuda().float()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h