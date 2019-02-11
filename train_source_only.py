import argparse
import os
import os.path as osp
import utils
import shutil
import time
import yaml
import pandas as pd
import numpy as np
import numpy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import logging
import se_transform.transforms as se_transforms
from dataset import NormalDataset, TeacherDataset
from utils import create_logger, AverageMeter, accuracy_2, save_checkpoint, load_state, IterLRScheduler
from utils import accuracy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--resume-opt', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--port', default='23456', type=str)


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(means=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)
    torch.cuda.manual_seed(int(time.time())%1000)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('inception_v3'):
        print('inception_v3 without aux_logits!')
        image_size = 341
        input_size = 299
        model = models.__dict__[args.arch](aux_logits=True,num_classes = 1000, pretrained = args.pretrained)
    else:
        image_size = 182
        input_size = 160
        student_model = models.__dict__[args.arch](num_classes = args.num_classes,
                                                   pretrained = args.pretrained,
                                                   avgpool_size=input_size/32)
    student_model.cuda()
    student_params = list(student_model.parameters())


    args.save_path = "checkpoint/" + args.exp_name

    if not osp.exists(args.save_path):
        os.mkdir(args.save_path)

    tb_logger = SummaryWriter(args.save_path)
    logger = create_logger('global_logger', args.save_path+'/log.txt')

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))


    logger.info("filename {}".format(osp.basename(__file__)))
    df_train = pd.read_csv(args.train_source_source, sep=" ", header=None)
    weight = df_train[1].value_counts()
    weight = weight.sort_index()
    weight = len(df_train) / weight
    weight = torch.from_numpy(weight.values).float().cuda()
    weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
    weight = (1 - torch.mean(weight)) + weight
    # define loss function (criterion) and optimizer
    if not args.use_weight:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    ignored_params = list(map(id, student_model.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad==True,
                         student_model.parameters())
    if args.pretrained==True:
        student_optimizer = torch.optim.Adam([
                                    {'params': base_params},
                                    {'params': student_model.classifier.parameters(), 'lr': args.base_lr}
                                    ] ,args.base_lr*0.1)
    else:
        student_optimizer = torch.optim.Adam(student_model.parameters(),
                                     args.base_lr)
    # optionally resume from a checkpoint
    print("Build network")
    last_iter = -1
    best_prec1 = 0
    if args.load_path:
        print(args.load_path)
        if args.resume_opt:
            best_prec1, last_iter = load_state(args.load_path, model, optimizer=student_optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    se_normalize = se_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    src_aug = se_transforms.ImageAugmentation(
        args.src_hflip, args.src_xlat_range,
        args.src_affine_std, rot_std = args.src_rot_std,
        intens_scale_range_lower = args.src_intens_scale_range_lower,
        intens_scale_range_upper = args.src_intens_scale_range_upper,
        colour_rot_std=args.src_colour_rot_std,
        colour_off_std=args.src_colour_off_std,
        greyscale=args.src_greyscale,
        scale_u_range=args.src_scale_u_range,
        scale_x_range=(None, None),
        scale_y_range=(None, None),
        cutout_probability = args.src_cutout_prob,
        cutout_size = args.src_cutout_size
    )
    tgt_aug = se_transforms.ImageAugmentation(
        args.tgt_hflip, args.tgt_xlat_range,
        args.tgt_affine_std, rot_std = args.tgt_rot_std,
        intens_scale_range_lower=args.tgt_intens_scale_range_lower,
        intens_scale_range_upper=args.tgt_intens_scale_range_upper,
        colour_rot_std=args.tgt_colour_rot_std,
        colour_off_std=args.tgt_colour_off_std,
        greyscale=args.tgt_greyscale,
        scale_u_range=args.tgt_scale_u_range,
        scale_x_range=[None, None],
        scale_y_range=[None, None],
        cutout_probability=args.tgt_cutout_prob,
        cutout_size=args.tgt_cutout_size)





    border_value = int(np.mean([0.485, 0.456, 0.406]) * 255 + 0.5)
    test_aug = se_transforms.ImageAugmentation(
        args.tgt_hflip, args.tgt_xlat_range,
        0.0, rot_std=0.0,
        scale_u_range=args.tgt_scale_u_range,
        scale_x_range=[None, None],
        scale_y_range=[None, None])

    train_source_dataset = NormalDataset(
        args.train_source_root,
        args.train_source_source,
        transform = transforms.Compose([
            se_transforms.ScaleCropAndAugmentAffine(
                (input_size, input_size),
                args.padding, True,
                src_aug, border_value,
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]))
        ]), args=args )

    train_target_dataset = TeacherDataset(
        args.train_target_root,
        args.train_target_source,
        transform = transforms.Compose([
            se_transforms.ScaleCropAndAugmentAffinePair(
                (input_size, input_size),
                args.padding, 0,
                True, tgt_aug, border_value,
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]))
        ]), args=args)

    val_dataset = NormalDataset(
        args.val_root,
        args.val_source,
        transform = transforms.Compose([
            se_transforms.ScaleAndCrop(
                (input_size, input_size),
                args.padding, False,
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]))
        ]),is_train=False, args=args )

    train_source_loader = DataLoader(
        train_source_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    train_target_loader = DataLoader(
        train_target_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, args.lr_steps, args.lr_gamma)
        #logger.info('{}'.format(args))
    if args.evaluate:
        validate(val_loader, student_model, criterion)
        return

    train(train_source_loader, train_target_loader, val_loader,
           student_model, criterion,
           student_optimizer = student_optimizer,
           lr_scheduler = lr_scheduler,
           start_iter = last_iter+1, tb_logger = tb_logger)

def train(train_source_loader, train_target_loader, val_loader,
          student_model, criterion, student_optimizer,
           lr_scheduler, start_iter, tb_logger):

    global best_prec1

    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)
    losses = AverageMeter(10)
    top1 = AverageMeter(10)
    top5 = AverageMeter(10)
    losses_bal = AverageMeter(10)
    confs_mask_count = AverageMeter(10)
    losses_aug = AverageMeter(10)
    losses_entropy = AverageMeter(10)
    losses_cls_uk = AverageMeter(10)
    losses_aug_uk = AverageMeter(10)
    losses_bal_uk = AverageMeter(10)
    student_model.train()
    # switch to train mode


    logger = logging.getLogger('global_logger')
    criterion_bce = nn.BCELoss()
    criterion_uk =  nn.BCEWithLogitsLoss()
    end = time.time()
    eval_output = []
    eval_target = []
    eval_uk = []
    for i, (batch_source, batch_target) in enumerate(zip(train_source_loader, train_target_loader)):
        input_source, label_source = batch_source

        curr_step = start_iter + i

        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]
        # measure data loading time
        data_time.update(time.time() - end)


        label_source = Variable(label_source).cuda(async=True)
        input_source = Variable(input_source).cuda()

        # compute output for source data
        source_output, source_output2 = student_model(input_source)

        # measure accuracy and record loss
        softmax_source_output = F.softmax(source_output, dim=1)

        #loss for known class
        if args.double_softmax:
            loss = criterion(softmax_source_output, label_source)
        else:
            loss = criterion(source_output, label_source)

        #loss for unknown class
        #integrate loss_cls and loss_entropy
        #compute accuracy
        prec1, prec5 = accuracy(softmax_source_output.data, label_source, topk=(1, 5))

        losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())
        # compute gradient and do SGD step



        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # measure elapsed time

        if curr_step % args.print_freq == 0 :
            tb_logger.add_scalar('loss_train', losses.avg, curr_step)
            tb_logger.add_scalar('acc1_train', top1.avg, curr_step)
            tb_logger.add_scalar('acc5_train', top5.avg, curr_step)
            tb_logger.add_scalar('lr', current_lr, curr_step)
            print(args.exp_name)
            logger.info('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.6f}'.format(
                   curr_step, len(train_source_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1=top1, top5=top5,
                   lr=current_lr))

        if (curr_step+1)%args.val_freq == 0 :

            val_loss, prec1, prec5  = validate(val_loader, student_model, criterion)
            if not tb_logger is None:
                tb_logger.add_scalar('loss_val', val_loss, curr_step)
                tb_logger.add_scalar('acc1_val', prec1, curr_step)
                tb_logger.add_scalar('acc5_val', prec5, curr_step)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info(" * best val prec@1 {}".format(best_prec1))
            save_checkpoint({
                'step': curr_step,
                'arch': args.arch,
                'state_dict': student_model.state_dict(),
                'best_prec1': best_prec1,
                'student_optimizer' : student_optimizer.state_dict(),
            }, is_best, args.save_path+'/ckpt' )


def validate(val_loader, model, criterion):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)
    # switch to evaluate mode
    model.train(mode=False)

    logger = logging.getLogger('global_logger')
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, output1 = model(input_var)

        # measure accuracy and record loss
        softmax_output = F.softmax(output, dim=1)
        #loss for known class
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(softmax_output.data, target, topk=(1, 5))
        #losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    model.train(mode=True)

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()
