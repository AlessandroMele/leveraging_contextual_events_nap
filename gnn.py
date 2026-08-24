from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu, dropout
from torch_geometric.nn.dense.linear import Linear
from torch import cat


class SAGE_GNN_var_1_2(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.aggregator = parameters.get('aggregator')

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        self.convs = ModuleList()

        self.convs.append(SAGEConv(in_channels=self.input_size, out_channels=self.layers_size, aggr=self.aggregator))
        for i in range(self.num_hidden):
            self.convs.append(SAGEConv(in_channels=self.layers_size, out_channels=self.layers_size, aggr=self.aggregator))
        self.convs.append(SAGEConv(in_channels=self.layers_size, out_channels=self.layers_size, aggr=self.aggregator))

        self.lin1 = Linear(self.layers_size, self.layers_size)
        self.lin2 = Linear(self.layers_size, self.output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = relu(x)
            x = dropout(x, p=self.dropout, training=self.training)

        x = self.readout(x, batch)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        return out


class SAGE_GNN_var_3_4(Module):
    def __init__(self, data, parameters):
        super().__init__()
        prefix = data.get('prefix')
        y = data.get('y')
        self.input_size = prefix.x.shape[-1]
        self.output_size = y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.aggregator = parameters.get('aggregator')

        self.context_size = int(self.layers_size/2)
        self.linear_size = self.layers_size + self.context_size
        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        self.convs = ModuleList()

        self.convs.append(SAGEConv(in_channels=self.input_size, out_channels=self.layers_size, aggr=self.aggregator))
        for i in range(self.num_hidden):
            self.convs.append(SAGEConv(in_channels=self.layers_size, out_channels=self.layers_size, aggr=self.aggregator))
        self.convs.append(SAGEConv(in_channels=self.layers_size, out_channels=self.layers_size, aggr=self.aggregator))

        self.context = SAGEConv(in_channels=self.input_size, out_channels=self.context_size, aggr=self.aggregator)

        self.lin1 = Linear(self.linear_size, self.linear_size)
        self.lin2 = Linear(self.linear_size, self.output_size)

    def forward(self, prefix, context):
        x, edge_index, batch = prefix.x, prefix.edge_index, prefix.batch
        c_x, c_edge_index, c_batch = context.x, context.edge_index, context.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = relu(x)
            x = dropout(x, p=self.dropout, training=self.training)

        c_x = self.context(c_x, c_edge_index)
        c_x = relu(c_x)
        c_x = dropout(c_x, p=self.dropout, training=self.training)

        x = self.readout(x, batch)
        c_x = self.readout(c_x, c_batch)
        x_concat = cat([x, c_x], dim=-1)

        x_concat = self.lin1(x_concat)
        x_concat = relu(x_concat)
        x_concat = dropout(x_concat, p=self.dropout, training=self.training)
        out = self.lin2(x_concat)

        return out
