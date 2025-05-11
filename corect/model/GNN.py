import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv

import corect

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
        super(GNN, self).__init__()
        self.args = args

        self.num_modals = num_modals
        
        if args.gcn_conv == "rgcn":
            print("GNN --> Use RGCN")
            self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)

        if args.use_graph_transformer:
            print("GNN --> Use Graph Transformer")
           
            in_dim = h1_dim
                
            self.conv2 = TransformerConv(in_dim, h2_dim, heads=args.graph_transformer_nheads, concat=True)
            self.bn = nn.BatchNorm1d(h2_dim * args.graph_transformer_nheads)
            

    def forward(self, node_features, node_type, edge_index, edge_type):

        if self.args.gcn_conv == "rgcn":
            x = self.conv1(node_features, edge_index, edge_type)
        
        if self.args.use_graph_transformer:
            x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index)))
        
        return x
        
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv,SAGEConv
# import torch.nn.functional as F


# import corect
# class GNN(nn.Module):
#     def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
#         super(GNN, self).__init__()
#         self.args = args
#         self.num_modals = num_modals

#         print("GNN --> Use Parallel GCN + GraphSAGE")

#         self.gcn = GCNConv(g_dim, h1_dim)
#         self.sage = SAGEConv(g_dim, h1_dim)
#         self.bn1 = nn.BatchNorm1d(h1_dim * 2)

#         self.conv2 = nn.Linear(h1_dim * 2, h2_dim)
#         self.bn2 = nn.BatchNorm1d(h2_dim)

#     def forward(self, node_features, node_type, edge_index, edge_type=None):
#         x1 = self.gcn(node_features, edge_index)
#         x2 = self.sage(node_features, edge_index)

#         x = torch.cat([x1, x2], dim=-1)  # Combine features
#         x = F.leaky_relu(self.bn1(x))

#         x = self.conv2(x)
#         x = F.leaky_relu(self.bn2(x))

#         return x

