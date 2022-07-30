import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from config import args_info, device
from models import CF,CF2
from utils import DataLoader, BenchmarkDataset
from utils.tools import Timer, model_save, get_lr, model_size, code_snapshot


def train(model, optimizer, train_loader, epoch, tb):
    model.train()  #首先指定了模型是什么，并指定训练
    alpha=1e-4
    for i, train_batch in enumerate(train_loader):
        train_batch = train_batch.to(device)

        ret,ret2,loss,batch_acc=model(train_batch)

        optimizer.zero_grad()
        loss.backward()  #更新模型
        optimizer.step()  #更新所有的参数，更新方法是optimizer
        if not (i % 20):
            print(f"Epoch[{epoch + 1}][{i + 1}/{len(train_loader)}] "
                  f"loss: {loss.item():.5f} "
                  f"acc: {batch_acc:.2f} ")

            iterations = epoch * len(train_loader) + i

            tb.add_scalar('data/tot_loss', loss.item(), iterations)
            tb.add_scalar('data/train_acc', batch_acc, iterations)
            tb.add_scalar('data/lr', get_lr(optimizer), iterations)

#修改后的inference阶段
def inference(model, auc_loader, fitb_loader, epoch, tb,c):  #inference需要减去category的y，考虑在model的测试阶段返回两部分分别的预测值，相减得到。
    model.eval()
    with torch.no_grad():
        # scores, labels ,scores_branch= [], [],[]
        scores, labels = [], []
        for auc_batch in auc_loader:
            auc_batch = auc_batch.to(device)
            auc_score = model.test_auc(auc_batch,c)  #这里需要更改
            scores.append(auc_score.cpu())
            # scores_branch.append(auc_score_branch.cpu())
            labels.append(auc_batch.y.cpu())



        scores = torch.cat(scores).numpy()
        # scores_branch=torch.cat(scores_branch).numpy()
        labels = torch.cat(labels).numpy()
        cp_auc = roc_auc_score(labels, scores)   #scores是预测结果  labels是给定的标签
        # cp_auc_branch=roc_auc_score(labels, scores_branch)

        fitb_right = 0
        # fitb_right_branch=0
        for fitb_batch in fitb_loader:
            fitb_batch = fitb_batch.to(device)
            tmp_fitb_right= model.test_fitb(fitb_batch,c)
            fitb_right+=tmp_fitb_right
            # fitb_right_branch+=tmp_fitb_right_branch
        fitb_acc = fitb_right / len(fitb_loader.dataset)
        # fitb_acc_branch=fitb_right_branch/len(fitb_loader.dataset)

    tb.add_scalars('data/kpi', {
        'auc': cp_auc,
        'fitb': fitb_acc
    }, epoch)

    total = cp_auc + fitb_acc

    return cp_auc, fitb_acc, total   #total帮助选择保留下效果最优的模型
def main(args):
    # tensorboard logging
    # web command: tensorboard --port=8097 --logdir tb_logs/exp1/ --bind_all
    tb = SummaryWriter(f'tb_logs/{args.remark}')

    # data preparation
    BenchmarkDataset.init(args)  #根据args生成benchmaekDataset
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_worker,
                     'pin_memory': True if torch.cuda.is_available() else False}
    train_dataset = BenchmarkDataset('train').next_train_epoch()  #
    auc_dataset_valid = BenchmarkDataset('valid').test_auc()
    fitb_dataset_valid = BenchmarkDataset('valid').test_fitb()
    auc_loader_valid = DataLoader(auc_dataset_valid, **loader_kwargs)
    fitb_loader_valid = DataLoader(fitb_dataset_valid, **loader_kwargs)

    # define model
    model = CF2(num_node_features=args.hid, num_cotpye=train_dataset.num_cotpye,
               depth=args.gd, nhead=args.nhead, dim_feedforward=args.fdim,
               num_layers=arg.nlayer, num_category=train_dataset.num_semantic_category).to(device)


    # define optimize and adjust lr
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)   #使用adam方式 及1e-5的learning rate 进行学习
    ajlr = ExponentialLR(optimizer, gamma=1 - 0.015)   #

    # model size
    print(f'  + Size of params: {model_size(model):.2f}MB')

    # record best result
    best_auc, best_fitb, best_total = 0., 0., 0.
    for epoch in range(args.epoch):
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)   #
        Timer.start('epoch_turn')

        # train for one epoch
        train(model, optimizer, train_loader, epoch, tb)   #train用的是
        # evaluate on validation set
        cp_auc, fitb_acc, kpi_total = inference(model, auc_loader_valid, fitb_loader_valid, epoch, tb,c=0)   #训练阶段的验证不需要减去c
        # update learning rate
        ajlr.step()
        #保留的是效果最好的模型
        #对比acc，留下acc最大的模型
        # remember best acc and save checkpoint
        is_best = kpi_total > best_total   #为什么留下
        best_auc = max(best_auc, cp_auc)
        best_fitb = max(best_fitb, fitb_acc)
        best_total = max(best_total, kpi_total)

        best_path = model_save(args.remark, model, epoch, is_best, best_auc=cp_auc, best_fitb=fitb_acc)   #remarks是时间  没有给fn 所以保存的是当前模型

        print(
            f"Epoch[{epoch + 1}][{Timer.end_time('epoch_turn'):.2f}s] "
            f"auc: {cp_auc:.4f} fitb: {fitb_acc:.4f} "
            f"best_auc: {best_auc:.4f} best_fitb: {best_fitb:.4f} ")

        # generate new negative training samples
        print('next epoch')
        train_dataset.next_train_epoch()

    # inference on test
    print('Train End!')
    auc_loader_test = DataLoader(BenchmarkDataset('test').test_auc(), **loader_kwargs)
    fitb_loader_test = DataLoader(BenchmarkDataset('test').test_fitb(), **loader_kwargs)

    ckpt = torch.load(best_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt["state_dict"])  #加载模型参数
    for k in range(0,21):
        c=k*0.05   #保证c从0~3，按0.1为间隔进行取值
        cp_auc, fitb_acc, _ = inference(model, auc_loader_test, fitb_loader_test, epoch, tb,c)   #加载完后直接用该参数进行预测即可
        print(f"c:{c}  auc: {cp_auc:.4f} fitb: {fitb_acc:.4f} ")

    tb.close()


if __name__ == '__main__':
    arg = args_info()
    code_snapshot(arg.remark)
    main(arg)

#首先根据train来得到类别间的共现次数（参考现有的共现矩阵生成代码）
#找一个阈值，并计算阈值上能够匹配的的套装的compatibility score和阈值下能够匹配的套装的compatibility score
#研究一下 compatibility modeling怎么得到AUC