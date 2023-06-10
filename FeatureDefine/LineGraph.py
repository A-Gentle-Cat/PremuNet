import networkx as nx
import torch


def get_line_graph(edge_index, cur_node_feature, cur_edge_feature):
    node_dict = {}
    line_node_dict = {}
    node_num = 0

    row, col = edge_index
    for i, X in enumerate(zip(row, col)):
        node_dict[(X[0].item(), X[1].item())] = i
    edges = [(min(X[0].item(), X[1].item()), max(X[0].item(), X[1].item()), {'idx': i}) for i, X in
             enumerate(zip(row, col))]
    G = nx.Graph(edges)
    H = nx.line_graph(G)
    H.add_nodes_from((node, G.edges[node]) for node in H)

    line_edge_index = []
    line_node_feature = []
    line_edge_feature = []

    if H.edges() == 0:
        return None

    for bond_from, bond_to in H.edges():
        if line_node_dict.get(bond_from) is None:
            line_node_dict[bond_from] = node_num
            node_num += 1
            idx = H.nodes[bond_from]['idx']
            line_node_feature.append(cur_edge_feature[idx])
        if line_node_dict.get(bond_to) is None:
            line_node_dict[bond_to] = node_num
            node_num += 1
            idx = H.nodes[bond_to]['idx']
            line_node_feature.append(cur_edge_feature[idx])
        line_edge_index.append([line_node_dict[bond_from], line_node_dict[bond_to]])
        line_edge_index.append([line_node_dict[bond_to], line_node_dict[bond_from]])

        idxs = sorted([bond_from[0], bond_from[1], bond_to[0], bond_to[1]])
        res_idx = 0
        for i in range(1, len(idxs)):
            if idxs[i] == idxs[i - 1]:
                res_idx = idxs[i]
                break

        line_edge_feature.append(cur_node_feature[res_idx])
        line_edge_feature.append(cur_node_feature[res_idx])

    if len(line_edge_feature) == 0:
        return None

    line_edge_index = torch.tensor(line_edge_index).t()
    line_node_feature = torch.stack(line_node_feature)
    line_edge_feature = torch.stack(line_edge_feature)

    return line_edge_index, line_node_feature, line_edge_feature


def get_line_graph2(edge_index, cur_node_feature, cur_edge_feature):
    line_edge_X = []
    line_edge_Y = []
    line_edge_feature = []

    line_node_feature = cur_edge_feature  # 将原图中边的feature直接变成线图中点的feature
    edges = []  # 将边表示为二元组
    edges_index = []
    edges_map = {}

    for i in range(len(edge_index[0])):
        edges.append((edge_index[0][i], edge_index[1][i]))
    edges.sort()  # 默认按照第一个元素排序

    num = 0
    for i in range(len(edges)):
        if (edges[i][1], edges[i][0]) in edges_map.keys():
            edges_index.append(edges_map[(edges[i][1], edges[i][0])])
        else:
            edges_map[edges[i]] = num
            edges_index.append(num)
            num += 1

    now = []
    now_index = 0
    # now_index表示中心点
    # now存储now_index的邻域
    for i in range(len(edges)):
        if i != 0 and edges[i][0] != edges[i - 1][0]:
            for u in now:
                for v in now:
                    if u != v:
                        line_edge_X.append(u)
                        line_edge_Y.append(v)
                        line_edge_feature.append(cur_node_feature[edges[now_index][0]])
            now.clear()
        now.append(edges_index[i])
        now_index = edges[i][0]

    # 处理完now目前剩余的点
    for u in now:
        for v in now:
            if u != v:
                line_edge_X.append(u)
                line_edge_Y.append(v)
                line_edge_feature.append(cur_node_feature[edges[now_index][0]])
    now.clear()

    line_edge_index = [line_edge_X, line_edge_Y]

    # print("line_edge_index: ", line_edge_index)
    # print("line_node_feature: ", line_node_feature)
    # print("line_edge_feature: ", line_edge_feature)

    return torch.tensor(line_edge_index), torch.tensor(line_node_feature), torch.stack(line_edge_feature)
