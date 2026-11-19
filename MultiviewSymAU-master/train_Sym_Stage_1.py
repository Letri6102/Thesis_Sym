import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.SymStage1 import MEFARGStage1
from model.contrastive import (
    MultiLabelSupConLoss,
    CrossModalAUExprLoss,
    EmbeddingQueue,
)
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)

def au_to_expr_pseudo(Y_a: torch.Tensor,
                      M_AE: torch.Tensor,
                      neutral_index: int) -> torch.Tensor:
    """
    Y_a: (B, N_a)  - nhãn AU (0/1) hoặc xác suất sau sigmoid
    M_AE: (N_a, N_e) - ma trận AU-Expression
    neutral_index: int - chỉ số class 'Neutral' trong trục expression

    return: Y_e (B, N_e), one-hot pseudo-label expression
    """
    B, N_a = Y_a.shape
    N_e = M_AE.shape[1]

    # Đảm bảo dùng float cho nhân ma trận
    Y_a_float = Y_a.float()

    # (1) Tính score theo Eq.(1): s = Y^a * M_{A-E}
    scores = Y_a_float @ M_AE    # (B, N_e)

    # (2) Chọn ke = argmax(score) cho mỗi sample
    ke = scores.argmax(dim=1)    # (B,)

    # (3) Xử lý các mẫu không có AU nào kích hoạt → Neutral
    neutral_mask = (Y_a_float.sum(dim=1) == 0)   # (B,)
    ke[neutral_mask] = neutral_index

    # (4) Tạo one-hot Y^e theo Eq.(2)
    Y_e = torch.zeros(B, N_e, device=Y_a.device, dtype=Y_a_float.dtype)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)

    return Y_e  # shape (B, N_e)

M_AE_np = np.load(r"matrixMAE\M_AE_DISFA.npy")
M_AE = torch.from_numpy(M_AE_np).float()

def contrastive_ramp(conf, epoch, batch_idx, num_batches):
    """Hệ số ramp tuyến tính từ 0 lên 1 trong `con_warmup_epochs` epoch đầu.

    Contrastive trên embedding còn ngẫu nhiên ở những bước đầu là nhiễu thuần
    tuý và có thể kéo backbone đi sai hướng trước khi nó học được gì. Ramp giữ
    cho L_wa dẫn dắt giai đoạn đầu.
    """
    warm = int(getattr(conf, "con_warmup_epochs", 0))
    if warm <= 0:
        return 1.0
    done = epoch * num_batches + batch_idx
    total = warm * num_batches
    return min(1.0, done / max(total, 1))


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion, criterion_Em,
          criterion_con=None, criterion_cross=None, queue=None):
    losses = AverageMeter()
    wa_meter = AverageMeter()
    we_meter = AverageMeter()
    con_meter = AverageMeter()
    cross_meter = AverageMeter()

    net.train()
    train_loader_len = len(train_loader)
    use_con = criterion_con is not None or criterion_cross is not None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (inputs, targets) in enumerate(pbar):

        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate,
                             batch_idx, train_loader_len)

        targets = targets.float()

        if torch.cuda.is_available():
            inputs  = inputs.cuda()
            targets = targets.cuda()

        # ----- Pseudo Label -----
        targets_Emo = au_to_expr_pseudo(targets, M_AE, neutral_index=6)
        if torch.cuda.is_available():
            targets_Emo = targets_Emo.cuda()

        optimizer.zero_grad()

        # ----- Forward -----
        if use_con:
            V_a, V_e, outputs_AU, outputs_Emo, z_g, z_a, z_e = net(inputs, return_proj=True)
        else:
            V_a, V_e, outputs_AU, outputs_Emo = net(inputs)
            z_g = z_a = z_e = None

        # ----- Loss gốc -----
        L_wa = criterion(outputs_AU, targets)              # AU loss (3)
        L_we = criterion_Em(outputs_Emo, targets_Emo)      # Emotion loss (4)

        gamma = conf.lam  # paper: gamma=0.05 (Section 4.1)
        loss = L_wa + gamma * L_we

        # ----- Contrastive -----
        L_con = torch.zeros((), device=loss.device)
        L_cross = torch.zeros((), device=loss.device)

        if use_con:
            ramp = contrastive_ramp(conf, epoch, batch_idx, train_loader_len)

            if criterion_con is not None:
                if queue is not None:
                    kf, kl = queue.get()
                else:
                    kf = kl = None
                L_con = criterion_con(z_g, targets, kf, kl)
                loss = loss + ramp * conf.con_weight * L_con

            if criterion_cross is not None:
                L_cross = criterion_cross(z_a, z_e, targets, M_AE)
                loss = loss + ramp * conf.cross_weight * L_cross

        # ----- Backprop -----
        loss.backward()
        optimizer.step()

        # Nạp queue SAU khi cập nhật, bằng embedding đã detach.
        if queue is not None and z_g is not None:
            queue.enqueue(z_g, targets)

        bs = inputs.size(0)
        losses.update(loss.item(), bs)
        wa_meter.update(L_wa.item(), bs)
        we_meter.update(L_we.item(), bs)
        con_meter.update(float(L_con.item()), bs)
        cross_meter.update(float(L_cross.item()), bs)

        # ====== TQDM cập nhật mô tả ======
        post = {
            'L_wa': f"{L_wa.item():.4f}",
            'L_we': f"{L_we.item():.4f}",
            'L_jf': f"{loss.item():.4f}",
        }
        if use_con:
            post['L_con'] = f"{con_meter.avg:.4f}"
            post['L_x'] = f"{cross_meter.avg:.4f}"
        pbar.set_postfix(post)

    return losses.avg, wa_meter.avg, we_meter.avg, con_meter.avg, cross_meter.avg



# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    emo_statistics_list = None
    emo_correct = 0
    emo_total = 0

    pbar = tqdm(val_loader, desc="Val")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            V_a, V_e, outputs_AU, outputs_Emo = net(inputs)

            # AU loss + statistics
            loss = criterion(outputs_AU, targets)
            losses.update(loss.item(), inputs.size(0))
            update_list = statistics(outputs_AU, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

            # Emotion statistics (pseudo labels)
            targets_Emo = au_to_expr_pseudo(targets, M_AE, neutral_index=6)
            emo_update = statistics(outputs_Emo, targets_Emo, 0.5)
            emo_statistics_list = update_statistics_list(emo_statistics_list, emo_update)
            emo_correct += (outputs_Emo.argmax(dim=1) == targets_Emo.argmax(dim=1)).sum().item()
            emo_total += targets.size(0)

            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    emo_mean_f1, emo_f1_list = calc_f1_score(emo_statistics_list)
    emo_acc = emo_correct / emo_total

    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list, emo_mean_f1, emo_acc

from datetime import datetime

def main(conf):


    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', conf.dataset+'_weight_fold'+str(conf.fold)+'.txt')))

    weight_path = os.path.join(conf.dataset_path, 'list',
                           f'{conf.dataset}_train_weight_fold{conf.fold}.txt')  # chú ý: thêm 'train_'
    train_weight = np.loadtxt(weight_path)
    train_weight = torch.from_numpy(train_weight).float()
    # 🔍 Kiểm tra trọng số WAL
    print(f"[WAL] weight_path = {weight_path}")
    print(f"[WAL] w = {train_weight.tolist()}, sum={float(train_weight.sum()):.6f}")

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    use_con = (conf.con_weight > 0) or (conf.cross_weight > 0)
    net = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7,
                       use_contrastive=use_con, proj_dim=conf.con_proj_dim)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
        global M_AE
        M_AE = M_AE.cuda()

    criterion = WeightedAsymmetricLoss(weight=train_weight)
    criterion_Em = ExpressionBCELoss()

    # ---------- Contrastive ----------
    criterion_con = None
    criterion_cross = None
    queue = None

    if conf.con_weight > 0:
        criterion_con = MultiLabelSupConLoss(
            temperature=conf.con_temp,
            neutral_weight=conf.con_neutral_weight,
            pos_threshold=conf.con_pos_threshold,
        )
        if conf.con_queue > 0:
            queue = EmbeddingQueue(dim=conf.con_proj_dim,
                                   num_aus=conf.num_classes,
                                   size=conf.con_queue)
            if torch.cuda.is_available():
                queue = queue.cuda()

    if conf.cross_weight > 0:
        criterion_cross = CrossModalAUExprLoss(
            temperature=conf.con_temp,
            target_temperature=(conf.con_target_temp if conf.con_target_temp > 0 else None),
        )

    logging.info(
        f"[CON] con_weight={conf.con_weight} cross_weight={conf.cross_weight} "
        f"tau={conf.con_temp} queue={conf.con_queue} proj_dim={conf.con_proj_dim} "
        f"neutral_w={conf.con_neutral_weight} warmup={conf.con_warmup_epochs} "
        f"target_temp={conf.con_target_temp}"
    )

    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    start_time = datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    best_f1 = 0.0
    #train and val
    for epoch in range(start_epoch, conf.epochs):

        print("--Weight will be Saved At", os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

        logging.info(f"[CFG] arc={conf.arc} K={conf.neighbor_num} metric={conf.metric} "
                 f"bs={conf.batch_size} lr0={conf.learning_rate} wd={conf.weight_decay}")

        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss, tr_wa, tr_we, tr_con, tr_cross = train(
            conf, net, train_loader, optimizer, epoch, criterion, criterion_Em,
            criterion_con=criterion_con, criterion_cross=criterion_cross, queue=queue)

        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc, val_emo_f1, val_emo_acc = val(net, val_loader, criterion)

        # log
        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  AU_F1 {:.2f}  AU_Acc {:.2f}  Emo_F1 {:.2f}  Emo_Acc {:.2f}'
                .format(epoch + 1, train_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc, 100.* val_emo_f1, 100.* val_emo_acc)}
        
        print(infostr)

        logging.info(infostr)
        logging.info('  [loss] L_wa: {:.5f}  L_we: {:.5f}  L_con: {:.5f}  L_cross: {:.5f}'
                     .format(tr_wa, tr_we, tr_con, tr_cross))
        logging.info('  [F1 per AU] ' + str(dataset_info(val_f1_score)))
        # infostr = {'F1-score-list:'}
        # logging.info(infostr)
        # infostr = dataset_info(val_f1_score)
        # logging.info(infostr)
        # infostr = {'Acc-list:'}
        # logging.info(infostr)
        # infostr = dataset_info(val_acc)
        # logging.info(infostr)

        if val_mean_f1_score > best_f1:
            best_f1 = val_mean_f1_score
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'best_model_fold' + str(conf.fold) + '.pth'))
            logging.info(f"[BEST] Epoch {epoch+1}: F1={100.*best_f1:.2f}% -> best_model_fold{conf.fold}.pth")

        # save checkpoints
        if (epoch+1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

        # checkpoint = {
        #     'epoch': epoch,
        #     'state_dict': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        # torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))


        
    end_time = datetime.now()
    print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Calculate duration
    duration = end_time - start_time
    print("Duration:", duration)

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
