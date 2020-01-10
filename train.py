import adabound
import argparse
import os
import pandas as pd
import random
import sys
import time
import torch
import torch.optim as optim
import yaml

from addict import Dict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.loss_fn import ActionSegmentationLoss
from libs.checkpoint import save_checkpoint, resume
from libs.class_weight import get_class_weight
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.metric import ScoreMeter, AverageMeter
from libs.transformer import TempDownSamp, ToTensor
from utils.class_id_map import get_id2class_map, get_n_classes


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--resume', action='store_true',
                        help='Add --resume option if you start training from checkpoint.')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, epoch, config, device):
    losses = AverageMeter('Loss', ':.4e')

    # switch training mode
    model.train()

    for i, sample in enumerate(train_loader):

        x = sample['feature']
        t = sample['label']

        x = x.to(device)
        t = t.to(device)

        batch_size = x.shape[0]

        # compute output and loss
        output = model(x)

        if isinstance(output, list):
            loss = 0.0
            for out in output:
                loss += criterion(out, t, x)
        else:
            loss = criterion(output, t, x)

        # record loss
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion, config, device):
    losses = AverageMeter('Loss', ':.4e')
    scores = ScoreMeter(
        id2class_map=get_id2class_map(
            config.dataset, dataset_dir=config.dataset_dir),
        thresholds=config.thresholds
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in val_loader:
            x = sample['feature']
            t = sample['label']
            x = x.to(device)
            t = t.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)

            loss = criterion(output, t, x)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # measure pixel accuracy, mean accuracy, Frequency Weighted IoU, mean IoU, class IoU
            pred = output.data.max(1)[1].squeeze(0).cpu().numpy()
            gt = t.data.cpu().squeeze(0).numpy()
            scores.update(pred, gt)

    acc, edit_score, f1s = scores.get_scores()

    return losses.avg, acc, edit_score, f1s


def main():
    # argparser
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # cpu or cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training CNN is computationally expensive.')
        sys.exit(1)

    # Dataloader
    # Temporal downsampling is applied to only videos in 50Salads
    print("Dataset: {}\tSplit: {}".format(CONFIG.dataset, CONFIG.split))
    print(
        "Batch Size: {}\tNum in channels: {}\tNum Workers: {}"
        .format(CONFIG.batch_size, CONFIG.in_channel, CONFIG.num_workers)
    )

    downsamp_rate = 2 if CONFIG.dataset == '50salads' else 1

    train_data = ActionSegmentationDataset(
        CONFIG.dataset,
        transform=Compose([
            ToTensor(),
            TempDownSamp(downsamp_rate),
        ]),
        mode='trainval' if not CONFIG.param_search else 'training',
        split=CONFIG.split,
        dataset_dir=CONFIG.dataset_dir,
        csv_dir=CONFIG.csv_dir
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True if CONFIG.batch_size > 1 else False,
        collate_fn=collate_fn
    )

    # if you do validation to determine hyperparams
    if CONFIG.param_search:
        val_data = ActionSegmentationDataset(
            CONFIG.dataset,
            transform=Compose([
                ToTensor(),
                TempDownSamp(downsamp_rate),
            ]),
            mode='validation',
            split=CONFIG.split,
            dataset_dir=CONFIG.dataset_dir,
            csv_dir=CONFIG.csv_dir
        )

        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=CONFIG.num_workers
        )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    n_classes = get_n_classes(CONFIG.dataset, dataset_dir=CONFIG.dataset_dir)

    print('Multi Stage TCN will be used as a model.')
    print('stages: {}\tn_features: {}\tn_layers of dilated TCN: {}\tkernel_size of ED-TCN: {}'
          .format(CONFIG.stages, CONFIG.n_features, CONFIG.dilated_n_layers, CONFIG.kernel_size))
    model = models.MultiStageTCN(
        in_channel=CONFIG.in_channel,
        n_classes=n_classes,
        stages=CONFIG.stages,
        n_features=CONFIG.n_features,
        dilated_n_layers=CONFIG.dilated_n_layers,
        kernel_size=CONFIG.kernel_size
    )

    # send the model to cuda/cpu
    model.to(device)

    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov
        )
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            model.parameters(),
            lr=CONFIG.learning_rate,
            final_lr=CONFIG.final_lr,
            weight_decay=CONFIG.weight_decay
        )
    else:
        print('There is no optimizer which suits to your option.')
        sys.exit(1)

    # learning rate scheduler
    if CONFIG.scheduler == 'onplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience
        )
    else:
        scheduler = None

    # resume if you want
    columns = ['epoch', 'lr', 'train_loss']

    # if you do validation to determine hyperparams
    if CONFIG.param_groups:
        columns += ['val_loss', 'acc', 'edit']
        columns += ["f1s@{}".format(CONFIG.thresholds[i])
                    for i in range(len(CONFIG.thresholds))]

    begin_epoch = 0
    best_loss = 100
    log = pd.DataFrame(columns=columns)
    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            checkpoint = resume(
                CONFIG.result_path, model, optimizer, scheduler)
            begin_epoch, model, optimizer, best_loss, scheduler = checkpoint
            print('training will start from {} epoch'.format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")
        if os.path.exists(os.path.join(CONFIG.result_path, 'log.csv')):
            print('loading the log file...')
            log = pd.read_csv(os.path.join(CONFIG.result_path, 'log.csv'))
        else:
            print("there is no log file at the result folder.")
            print('Making a log file...')

    # criterion for loss
    if CONFIG.class_weight:
        class_weight = get_class_weight(
            CONFIG.dataset, split=CONFIG.split, csv_dir=CONFIG.csv_dir)
        class_weight = class_weight.to(device)
    else:
        class_weight = None

    criterion = ActionSegmentationLoss(
        ce=CONFIG.ce, tmse=CONFIG.tmse, weight=class_weight,
        ignore_index=255, tmse_weight=CONFIG.tmse_weight
    )

    # train and validate model
    print('\n---------------------------Start training---------------------------\n')
    for epoch in range(begin_epoch, CONFIG.max_epoch):
        # training
        start = time.time()
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch, CONFIG, device)
        train_time = (time.time() - start) / 60

        # if you do validation to determine hyperparams
        if CONFIG.param_search:
            start = time.time()
            val_loss, acc, edit_score, f1s = validate(
                val_loader, model, criterion, CONFIG, device)
            val_time = (time.time() - start) / 60

            # save a model if top1 acc is higher than ever
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_loss_model.prm')
                )

        # save checkpoint every epoch
        save_checkpoint(
            CONFIG.result_path, epoch, model, optimizer, best_loss, scheduler)

        # write logs to dataframe and csv file
        tmp = [epoch, optimizer.param_groups[0]['lr'], train_loss]

        # if you do validation to determine hyperparams
        if CONFIG.param_search:
            tmp += [val_loss, acc, edit_score]
            tmp += [f1s[-1][i] for i in range(len(CONFIG.thresholds))]

        tmp_df = pd.Series(tmp, index=log.columns)

        log = log.append(tmp_df, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        if CONFIG.param_search:
            # if you do validation to determine hyperparams
            print(
                'epoch: {}\tlr: {:.4f}\ttrain_time: {:.1f}min\tval_time: {:.1f}min\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc: {:.4f}\tval_edit: {:.4f}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, val_time,
                        train_loss, val_loss, acc, edit_score)
            )
        else:
            print(
                'epoch: {}\tlr: {:.4f}\ttrain_time: {:.1f}min\ttrain loss: {:.4f}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss)
            )

    # save models
    torch.save(
        model.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))

    print("Done!")
    print("")


if __name__ == '__main__':
    main()
