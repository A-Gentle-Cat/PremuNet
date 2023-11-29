import copy
import random
from typing import Tuple, Union, List

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import torch_geometric.data as tdata
from torch_geometric.loader.dataloader import Collater
from torchviz import make_dot
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from FeatureDefine.MolFeature import MolFeature
from LoadData.LoadDataset import molDataset
from Models.AblationModel import FPN_Tradi
from Models.FPN import FPN
from Models.FP_GNN_NET import FP_GNN_NET
from Models.ModelGetter import get_model
from Models.PNA import *
from Models.Uni_MPNN import Uni_Net
from UnifiedMolPretrain.pretrain3d.model.gnn import GNNet
from Utils.Accumulator import Accumulator
from Utils.PreProcessor import PreprocessBatch
from Utils.load_utils import get_split_idx
import config


def accuracy_calculator_sigmoid(output: torch.Tensor, target: torch.Tensor):
    output = torch.sigmoid(output)
    output = torch.where(output >= 0.5, 1, 0)
    total = 0
    cnt = 0
    for i in range(config.task_num):
        cur_out = output[:, i]
        cur_tar = target[:, i]
        use_idx = cur_tar == cur_tar
        cur_out = cur_out[use_idx]
        cur_tar = cur_tar[use_idx]

        res = (cur_out == cur_tar)
        if len(cur_out) == 0:
            continue
        cnt += 1
        total += res.sum().item() / len(cur_out)
    return total / cnt


def calcul_auc(output, target: torch.Tensor, catogory='train') -> float:
    output = torch.sigmoid(output)
    res_auc = 0
    for i in range(config.task_num):
        try:
            cur_out = output[:, i]
            cur_tar = target[:, i]
            use_idx = cur_tar == cur_tar
            cur_out = cur_out[use_idx]
            cur_tar = cur_tar[use_idx]
            res_auc += roc_auc_score(cur_tar.cpu().detach().numpy(), cur_out.cpu().detach().numpy())
        except Exception as e:
            if catogory != 'train':
                print('AUC ERROR', e)
            res_auc += 1.0
    return res_auc / config.task_num


def calcul_regression(output, target, task_num=None):
    if len(target.shape) != 2:
        target = target.unsqueeze(1)
    if len(output.shape) != 2:
        output = output.unsqueeze(1)
    # print(output.shape, target.shape)
    if task_num is None:
        task_num = config.task_num
    res_metrics = 0
    # print('task num =', task_num)
    for i in range(task_num):
        if config.valid_metrics == 'MSE':
            res_metrics += mean_squared_error(target[:, i].cpu().detach().numpy(), output[:, i].cpu().detach().numpy(),
                                              squared=True)
        elif config.valid_metrics == 'MAE':
            res_metrics += mean_absolute_error(target[:, i].cpu().detach().numpy(), output[:, i].cpu().detach().numpy())
        elif config.valid_metrics == 'RMSE':
            res_metrics += mean_squared_error(target[:, i].cpu().detach().numpy(), output[:, i].cpu().detach().numpy(),
                                              squared=False)
        else:
            raise Exception(f'not supported for metrics {config.valid_metrics}')
    return res_metrics / task_num


def evaluate(net: nn.Module, loss, eva_data):
    """
    @return: classification: （loss，acc，auc）；regression: （loss，metrics）
    """
    out_res = []
    target = []
    with torch.no_grad():
        counter = Accumulator(5) if config.dataset_type == 'classification' else Accumulator(4)
        task_counter = Accumulator(config.task_num + 1)
        for X in eva_data:
            y = torch.tensor(np.array(X.y)).to(config.device).to(torch.float32)
            y = y.reshape(-1, config.task_num)
            target.append(y)

            # out, out_gnn, out_sch = net(X)
            X = X.to(config.device)
            net.eval()
            out = net(X)
            if isinstance(out, Tuple):
                out = out[0]
            use_idx = y == y
            L = loss(out[use_idx], y[use_idx])
            out_res.append(out)

            if config.dataset_type == 'classification':
                counter.add(accuracy_calculator_sigmoid(out, y), len(y), L * len(y), len(y), 1.0)
            else:
                counter.add(calcul_regression(out, y), 1.0, L * (len(y)), len(y))

        out_res = torch.cat(out_res, dim=0)
        target = torch.cat(target, dim=0)
        # print(f'valid:: out_res.shape: {out_res} target.shape: {target.shape}')

        if config.dataset_type == 'classification':
            auc_res = calcul_auc(out_res, target, 'valid')
            return counter[2] / counter[3], counter[0] / counter[4], auc_res
        else:
            return counter[2] / counter[3], counter[0] / counter[1]


def draw_total(his: dict, dataset_name, num_step):
    plt.figure(dpi=300)

    num_photos = len(his)
    plt.suptitle(dataset_name + f' {num_step} Fold')
    for i, key in enumerate(his.keys()):
        plt.subplot(2, int(round(num_photos / 2, 0)), i + 1)
        x = range(1, len(his[key]) + 1)
        y = his[key]

        x_new = np.linspace(min(x), max(x), len(x) * 5)
        y_smooth = make_interp_spline(x, y)(x_new)

        plt.plot(x_new, y_smooth, linewidth=1)
        plt.xlabel('Epoch')
        plt.title(key)


def get_output_separate(net: Uni_Net, test_iter):
    out1 = [[], [], []]  # sum, mean, max
    out2 = [[], [], []]  # sum, mean, max
    for X in test_iter:
        cur1 = net.fcgnn_model(X).cpu().detach().numpy()
        cur2 = net.unified_model(X)[0].cpu().detach().numpy()

        out1[0].extend(np.sum(cur1, axis=1))
        out1[1].extend(np.mean(cur1, axis=1))
        out1[2].extend(np.max(cur1, axis=1))

        out2[0].extend(np.sum(cur2, axis=1))
        out2[1].extend(np.mean(cur2, axis=1))
        out2[2].extend(np.max(cur2, axis=1))
        # print(f'cur1: {out1[0].shape} cur2: {out2[0].shape}')
        # break
    for i in range(3):
        out1[i] = round(float(np.mean(out1[i])), 3)
        out2[i] = round(float(np.mean(out2[i])), 3)

    print(f'PremuNet-L: out_sum: {out1[0]} out_mean: {out1[1]} out_max: {out1[2]}')
    print(f'PremuNet-H: out_sum: {out2[0]} out_mean: {out2[1]} out_max: {out2[2]}')
    write_to_trainlog(f'PremuNet-L: out_sum: {out1[0]} out_mean: {out1[1]} out_max: {out1[2]}')
    write_to_trainlog(f'PremuNet-H: out_sum: {out2[0]} out_mean: {out2[1]} out_max: {out2[2]}')



def train(args, net, train_iter, valid_data, test_data, dataset_name, num_step=0) -> Union[
    Tuple[dict, float, float], Tuple[dict, float]]:
    if config.dataset_type == 'classification':
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.MSELoss()
    print('loss: ', loss)
    # loss = AUCMLoss_V1()
    # loss = nn.L1Loss()
    # loss = Sp_GNN_FP_Net_Loss()
    # universical
    if isinstance(net, FPN):
        updater = torch.optim.Adam(net.parameters(), lr=config.fcn_lr)
    elif isinstance(net, FP_GNN_NET):
        updater = torch.optim.Adam([{"params": net.fpn_model.parameters(), "lr": config.fcn_lr}], lr=config.gnn_lr)
    elif isinstance(net, GNNet):
        updater = torch.optim.Adam(net.parameters(), lr=config.unified_lr)
        # updater = torch.optim.Adam([{"params": net.gnn_layers.pos_embedding.pos_embedding_schnet.parameters(), "lr": 1e-3}], lr=config.unified_lr)
    elif isinstance(net, Uni_Net):
        updater = torch.optim.Adam([{"params": net.fcgnn_model.fpn_model.parameters(), "lr": config.fcn_lr},
                                    {"params": net.fcgnn_model.gnn_model.parameters(), "lr": config.gnn_lr},
                                    {"params": net.unified_model.parameters(), "lr": config.unified_lr}],
                                   lr=config.LR)
    else:
        updater = torch.optim.Adam(net.parameters(), lr=config.LR)
    # fp_gnn_net
    # scheduler = NoamLR(optimizer=updater, warmup_epochs=[warmup_epochs],
    #                    total_epochs=None or [EPOCH] * num_lrs,
    #                    steps_per_epoch=config.train_data_size // BATCH_SIZE, init_lr=[init_lr],
    #                    max_lr=[max_lr],
    #                    final_lr=[final_lr])
    scheduler = StepLR(updater, step_size=config.decay_step, gamma=config.decay_gamma)
    net = net.to(config.device)

    best_test_res = 0
    best_valid_res = 0
    test_his = []

    if config.dataset_type == 'classification':
        history = {'TrainLoss': [], 'TrainAcc': [], 'TrainAUC': [], 'ValidLoss': [], 'ValidAcc': [], 'ValidAUC': []}
    else:
        history = {'TrainLoss': [], f'Train{config.valid_metrics}': [], 'ValidLoss': [],
                   f'Valid{config.valid_metrics}': []}

    # for k, v in net.named_parameters():
    #     if 'pos_embedding_schnet' not in k:
    #         v.requires_grad = False

    for epoch in range(config.EPOCH):
        train_counter = Accumulator(5) if config.dataset_type == 'classification' else Accumulator(4)
        data_iter = tqdm(train_iter)
        for X in data_iter:
            updater.zero_grad()
            y = torch.tensor(np.array(X.y), dtype=torch.float32).to(config.device)
            y = y.reshape(-1, config.task_num)

            X = X.to(config.device)
            net.train()
            out = net(X)
            # print(f'max: {torch.max(out)} min: {torch.min(out)} mean: {torch.mean(out)}')
            if isinstance(out, Tuple):
                out = out[0]

            use_idx = y == y
            L = loss(out[use_idx], y[use_idx])
            L.backward()
            if config.grad_clip is not None:
                clip_grad_norm_(net.parameters(), max_norm=config.grad_clip, norm_type=2.0)
            updater.step()

            if config.dataset_type == 'classification':
                train_counter.add(float(L.item()) * len(y[use_idx]),
                                  accuracy_calculator_sigmoid(out, y),
                                  len(y[use_idx]),
                                  calcul_auc(out, y),
                                  1.0)
                train_res = [train_counter[0] / train_counter[2],
                             train_counter[1] / train_counter[4],
                             calcul_auc(out, y)]

                data_iter.set_description(
                    f'Epoch:{epoch + 1}/{config.EPOCH} '
                    f'TrainLoss: {round(train_res[0], 5)} '
                    f'TrainAcc: {round(train_res[1], 5)} '
                    f'TrainAUC: {round(train_res[2], 5)} '
                    f'lr：{round(scheduler.get_last_lr()[0], 7)}'
                )
            else:
                train_counter.add(float(L) * len(y[use_idx]), len(y[use_idx]), calcul_regression(out, y), 1.0)
                train_res = [train_counter[0] / train_counter[1],
                             train_counter[2] / train_counter[3]]

                data_iter.set_description(
                    f'Epoch:{epoch + 1}/{config.EPOCH} '
                    f'TrainLoss: {round(train_res[0], 5)} '
                    f'Train{config.valid_metrics}: {round(train_res[1], 5)} '
                    f'lr：{round(scheduler.get_last_lr()[0], 7)}'
                )

        scheduler.step()

        valid_res = evaluate(net, loss, valid_data)
        valid_res2 = evaluate(net, loss, valid_data)

        # print('==============================================')
        # print(valid_res[1], valid_res2[1])
        # print('==============================================')

        if config.dataset_type == 'classification':
            history['TrainLoss'].append(np.log(train_counter[0] / train_counter[2]))
            history['TrainAcc'].append(train_counter[1] / train_counter[4])
            history['TrainAUC'].append(train_counter[3] / train_counter[4])
            history['ValidLoss'].append(np.log(valid_res[0]))
            history['ValidAcc'].append(valid_res[1])
            history['ValidAUC'].append(valid_res[2])
        else:
            history['TrainLoss'].append(np.log(train_counter[0] / train_counter[1]))
            history[f'Train{config.valid_metrics}'].append(train_counter[2] / train_counter[3])
            history['ValidLoss'].append(np.log(valid_res[0]))
            history[f'Valid{config.valid_metrics}'].append(valid_res[1])

        if config.dataset_type == 'classification':
            write_to_trainlog(f'Epoch:{epoch + 1}/{config.EPOCH} '
                              f'TrainLoss: {round(train_counter[0] / train_counter[2], 5)} '
                              f'TrainAcc: {round(train_counter[1] / train_counter[2], 5)} '
                              f'TrainAUC: {round(train_counter[3] / train_counter[4], 5)} '
                              f'ValidLoss: {round(np.log(valid_res[0]), 5)} '
                              f'ValidAcc: {round(valid_res[1], 5)} '
                              f'ValidAUC: {round(valid_res[2], 5)} '
                              f'LR: {round(scheduler.get_last_lr()[0], 5)}')
        else:
            write_to_trainlog(f'Epoch: {epoch + 1}/{config.EPOCH} '
                              f'TrainLoss: {round(train_counter[0] / train_counter[1], 5)} '
                              f'Train{config.valid_metrics}: {round(train_counter[2] / train_counter[3], 5)} '
                              f'ValidLoss: {round(np.log(valid_res[0]), 5)} '
                              f'Valid{config.valid_metrics}: {round(valid_res[1], 5)} '
                              f'LR: {round(scheduler.get_last_lr()[0], 5)}')

        if config.dataset_type == 'classification':
            print(
                f' Valid Loss: {round(valid_res[0], 5)}  Valid ACC: {round(valid_res[1], 5)}  Valid AUC: {round(valid_res[2], 5)}')
            write_to_trainlog(
                f' Valid Loss: {round(valid_res[0], 5)}  Valid ACC: {round(valid_res[1], 5)}  Valid AUC: {round(valid_res[2], 5)}')
        else:
            print(
                f' Valid {config.valid_metrics}: {round(valid_res[1], 5)} lr：{round(scheduler.get_last_lr()[0], 7)}')
            write_to_trainlog(
                f' Valid {config.valid_metrics}: {round(valid_res[1], 5)} lr：{round(scheduler.get_last_lr()[0], 7)}')

        if test_data is not None:
            test_res = evaluate(net, loss, test_data)
            if config.dataset_type == 'classification':
                best_valid_res = max(history['ValidAUC'])
                best_test_res = max(best_test_res, test_res[2])
                print(f'Test Acc: {round(test_res[1], 5)} AUC: {round(test_res[2], 5)}')
                write_to_trainlog(f'Test Acc: {round(test_res[1], 5)} AUC: {round(test_res[2], 5)}')
                test_his.append(test_res[2])
            else:
                print(f'Test {config.valid_metrics}: {round(test_res[1], 5)}')
                write_to_trainlog(f'Test {config.valid_metrics}: {round(test_res[1], 5)}')

        # if isinstance(net, Uni_Net):
        #     get_output_separate(net, test_data)

    # draw_total(history, dataset_name, num_step)

    # if isinstance(net, Uni_Net):
    #     get_output_separate(net, test_data)

    # Test result
    np.save('unified_schnet_auc', np.array(test_his))

    test_res = evaluate(net, loss, test_data)
    if config.dataset_type == 'classification':
        best_valid_res = max(history['ValidAUC'])
        best_test_res = max(best_test_res, test_res[2])
        print(f'Test Acc: {round(test_res[1], 5)} AUC: {round(test_res[2], 5)}')
        write_to_trainlog(f'Test Acc: {round(test_res[1], 5)} AUC: {round(test_res[2], 5)}')
        test_his.append(test_res[2])
        return history, test_res[1], test_res[2]
    else:
        print(f'Test {config.valid_metrics}: {round(test_res[1], 5)}')
        write_to_trainlog(f'Test {config.valid_metrics}: {round(test_res[1], 5)}')
        return history, test_res[1]


def write_to_trainlog(*args):
    if config.train_logpath is None:
        return
    train_logpath = open(config.train_logpath, 'a+')
    for s in args:
        train_logpath.write(s + '\n')
    train_logpath.close()


def one_split_train(args, num_step, train_dataset, valid_dataset, test_datasset):
    print('================================================================== '
          f'Training num: [{num_step}] '
          '==================================================================')
    print(f'=========dataset: 【{args.dataset}】=========')
    print(f'unified weight path：{config.uni_checkpoint_path}')
    write_to_trainlog(f'unified weight path: {config.uni_checkpoint_path}')
    write_to_trainlog(f'raw_with_pos: {config.raw_with_pos}')
    write_to_trainlog(f'agg_method: {config.agg_method}')
    if config.use_fingerprints_file:
        config.fingerprints_file_name = args.dataset

    print(f'dataset: 【{args.dataset}】')
    print(f'train data length: {len(train_dataset)} valid data length: {len(valid_dataset)}')
    config.task_num = train_dataset.task_num
    print(f'Task Num: [{config.task_num}]')
    config.train_data_size = len(train_dataset)

    torch.manual_seed(seed=config.seed)
    g = torch.Generator()
    train_iter = tdata.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=Collater(None, None),
                                  shuffle=True,
                                  generator=g,
                                  drop_last=False)
    torch.manual_seed(seed=config.seed)
    g = torch.Generator()
    valid_iter = tdata.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, collate_fn=Collater(None, None),
                                  shuffle=False, generator=g, drop_last=False)
    torch.manual_seed(seed=config.seed)
    g = torch.Generator()
    test_iter = tdata.DataLoader(test_datasset, batch_size=config.BATCH_SIZE, collate_fn=Collater(None, None),
                                 shuffle=False, generator=g, drop_last=False)
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    config.deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        config.deg += torch.bincount(d, minlength=config.deg.numel())

    mynet = get_model(args.model)
    # print(mynet)

    print('Model Total parameters:', sum(p.numel() for p in mynet.parameters()))
    write_to_trainlog(f'Model Total Parameters: {sum(p.numel() for p in mynet.parameters())}')
    write_to_trainlog('============================================'
                      f' Training num: [{num_step}] '
                      '============================================',
                      f'=========dataset: 【{args.dataset}】=========',
                      f'train length: {len(train_dataset)} valid length: {len(valid_dataset)}',
                      f'use_rdkit_feature: {config.use_rdkit_feature}',
                      f'use_pretrained_atom_feature: {config.use_pretrained_atom_feature}',
                      f'Task Num: [{config.task_num}]',
                      f'BatchSize = {config.BATCH_SIZE}',
                      f'LR: gnn_lr = {config.gnn_lr} fcn_lr = {config.fcn_lr}',
                      f'Model type: \n【{type(mynet)}】')
    print(type(mynet))

    res = train(args, mynet, train_iter, valid_iter, test_iter, args.dataset,
                num_step=num_step)
    history = res[0]

    plt.tight_layout()
    photo_path = config.photo_path + f'_{num_step}.png'
    plt.savefig(photo_path, dpi=300)


    if config.dataset_type == 'classification':
        testAcc = res[1]
        testAUC = res[2]
        write_to_trainlog(f'TrainLoss: {history["TrainLoss"][-1]}\n'
                          f'TrainAcc: {history["TrainAcc"][-1]}\n'
                          f'TrainAUC: {history["TrainAUC"][-1]}\n'
                          f'ValidLoss: {history["ValidLoss"][-1]}\n'
                          f'ValidAcc: {history["ValidAcc"][-1]}\n'
                          f'ValidAUC: {history["ValidAUC"][-1]}\n'
                          f'TestACC: {testAcc}TestAUC: {testAUC}\n'
                          f'Train Graph: {photo_path}\n'
                          '===============================================================\n')
        return testAcc, testAUC, mynet
    else:
        testmetrics = res[1]
        write_to_trainlog(f'TrainLoss: {history["TrainLoss"][-1]}\n'
                          f'Train{config.valid_metrics}: {history["TrainLoss"][-1]}\n'
                          f'ValidLoss: {history["ValidLoss"][-1]}\n'
                          f'Valid{config.valid_metrics}: {history[f"Valid{config.valid_metrics}"][-1]}\n'
                          f'Test{config.valid_metrics}: {testmetrics}\n'
                          f'Train Graph: {photo_path}\n'
                          '===============================================================\n')
        return testmetrics, mynet


def reset_seed():
    if not config.keep_seed:
        config.seed = np.random.randint(0, 100000)
        print(f'Seed Reset! config.seed = 【{config.seed}】')
        write_to_trainlog(f'Seed Reset! config.seed = 【{config.seed}】')
    else:
        print(f'Seed keep! config.seed = 【{config.seed}】')
        write_to_trainlog(f'Seed keep config.seed = 【{config.seed}】')
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)


def start_split_train(args):
    testACC, testAUC = 0, 0
    testmetrics = 0

    bestAUC = 0
    bestmetrics = 0
    best_model = None

    fold_res_list = []

    for K in range(config.random_epoch):
        reset_seed()
        data = molDataset(args=args, dataset_name=args.dataset, catogory='all', reset=config.reset)
        data_idx = get_split_idx(data.data_dir, len(data))
        train_data = data[data_idx['train']]
        valid_data = data[data_idx['valid']]
        test_data = data[data_idx['test']]
        split_res = one_split_train(args, K + 1, train_data, valid_data, test_data)
        if config.dataset_type == 'classification':
            curACC, curAUC, mynet = split_res
            testACC += curACC
            testAUC += curAUC
            if curAUC > bestAUC:
                best_model = copy.deepcopy(mynet)
                bestAUC = curAUC
            fold_res_list.append(curAUC)
        else:
            curmetrics, mynet = split_res
            testmetrics += curmetrics
            if curmetrics < bestmetrics:
                best_model = copy.deepcopy(mynet)
                bestmetrics = curmetrics
            fold_res_list.append(curmetrics)

    if config.dataset_type != 'classification':
        for value in config.testTask:
            config.testTask[value] = value / config.random_epoch

    if config.dataset_type == 'classification':
        print(f'Split {config.random_epoch} result：{fold_res_list}')
        write_to_trainlog(f'Split {config.random_epoch} result: {fold_res_list}')
        print(f'Split [{config.random_epoch}] Test Best AUC: {bestAUC}')
        write_to_trainlog(f'Split {config.random_epoch} result： Best AUC: {bestAUC}')
        print(
            f'{config.random_epoch} Avg ACC: {testACC / config.random_epoch} Avg AUC: {testAUC / config.random_epoch} std: {np.std(fold_res_list)}')
        write_to_trainlog(
            f'{config.random_epoch} Avg ACC: {testACC / config.random_epoch} Avg AUC: {testAUC / config.random_epoch} std: {np.std(fold_res_list)}\n')
        if config.save_model:
            model_path = os.path.join(config.model_dir,
                                      f'{args.model}_Best_MeanAUC_{round(testAUC / config.random_epoch, 3)}.pkl')
            torch.save(best_model, model_path)
            write_to_trainlog(f'Model save to: {model_path}')
    else:
        print(f'Split {config.random_epoch} result：{fold_res_list}')
        write_to_trainlog(f'Split {config.random_epoch} result：{fold_res_list}')
        print(f'Split {config.random_epoch} result： Best {config.valid_metrics}: {bestmetrics}')
        write_to_trainlog(f'Split {config.random_epoch} result： Best {config.valid_metrics}: {bestmetrics}')
        print(
            f'{config.random_epoch} Avg Test: {config.valid_metrics}: {testmetrics / config.random_epoch} std: {np.std(fold_res_list)}')
        write_to_trainlog(
            f'{config.random_epoch} Avg Test: {config.valid_metrics}: {testmetrics / config.random_epoch} std: {np.std(fold_res_list)}')
        if config.save_model:
            model_path = os.path.join(config.model_dir,
                                      f'{args.dataset}_Best_MeanMAE_{round(testmetrics / config.random_epoch, 3)}.pkl')
            torch.save(best_model, model_path)
            write_to_trainlog(f'Model save to: {model_path}')

    print(f'log file save to: {config.train_logpath}')
    write_to_trainlog(f'log file save to: {config.train_logpath}')

    plt.show()
