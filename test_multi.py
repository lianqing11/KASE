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

    student_optimizer = torch.optim.Adam(student_model.parameters(), args.base_lr*0.1)

    args.save_path = "checkpoint/" + args.exp_name


    if not osp.exists(args.save_path):
        os.mkdir(args.save_path)

    tb_logger = SummaryWriter(args.save_path)
    logger = create_logger('global_logger', args.save_path+'/log.txt')

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))


    criterion = nn.CrossEntropyLoss()
    print("Build network")
    last_iter = -1
    best_prec1 = 0
    load_state(args.save_path + "/ckptmodel_best.pth.tar", student_model)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    se_normalize = se_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])






    border_value = int(np.mean([0.485, 0.456, 0.406]) * 255 + 0.5)
    test_aug = se_transforms.ImageAugmentation(
        True, 0, rot_std=0.0,
        scale_u_range=[0.75, 1.333],
        affine_std=0,
        scale_x_range=None,
        scale_y_range=None)

    val_dataset = NormalDataset(
        args.val_root,
        "./data/visda/list/validation_list.txt",
        transform = transforms.Compose([
            se_transforms.ScaleAndCrop(
                (input_size, input_size),
                args.padding, False,
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]))
        ]),is_train=False, args=args )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers)


    val_multi_dataset = NormalDataset(
        args.val_root,
        "./data/visda/list/validation_list.txt",
        transform = transforms.Compose([
            se_transforms.ScaleCropAndAugmentAffineMultiple(
                16,
                (input_size, input_size),
                args.padding, True, test_aug, border_value,
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]))
        ]),is_train=False, args=args )


    val_multi_loader = DataLoader(
        val_multi_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, args.lr_steps, args.lr_gamma)
        #logger.info('{}'.format(args))
    validate(val_loader, student_model, criterion)
    validate_multi(val_multi_loader, student_model, criterion)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)
    # switch to evaluate mode
    model.eval()
    eval_target = []
    eval_output = []
    eval_uk = []

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
        output1 = F.sigmoid(output1)
        eval_target.append(target.cpu().data.numpy())
        eval_output.append(softmax_output.cpu().data.numpy())
        eval_uk.append(output1.cpu().data.numpy())
        prec1, prec5 = accuracy(softmax_output.data, target, topk=(1, 5))
        #losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))


    eval_target = np.concatenate(eval_target, axis=0)
    eval_output = np.concatenate(eval_output, axis=0)
    eval_uk = np.concatenate(eval_uk, axis=0)
    evaluator = utils.PredictionEvaluator_2(eval_target, args.num_classes)
    for i in range(10):
        t_clss_acc, t_aug_cls_acc = evaluator.evaluate(eval_output, eval_uk, i*0.1)
        print("epslion {:.2f}, mean_aug_class_acc {}, aug_cls_acc {}".format(i*0.1, t_clss_acc, t_aug_cls_acc))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    model.train(mode=True)

    return losses.avg, top1.avg, top5.avg



def validate_multi(val_loader, model, criterion):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)
    # switch to evaluate mode
    model.eval()

    logger = logging.getLogger('global_logger')
    end = time.time()
    eval_output = []
    eval_target = []
    eval_uk = []
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, output1 = model(input_var.squeeze(0).float())


        # measure accuracy and record loss
        softmax_output = F.softmax(output, dim=1).mean(0)
        softmax_output = softmax_output.unsqueeze(0)
        output1 = F.sigmoid(output1).mean(0).unsqueeze(0)
        #loss for known class
        eval_output.append(softmax_output.cpu().data.numpy())
        eval_target.append(target_var.cpu().data.numpy())
        eval_uk.append(output1.cpu().data.numpy())
        prec1, prec5 = accuracy(softmax_output.data, target, topk=(1, 5))
        #losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    eval_target = np.concatenate(eval_target, axis=0)
    eval_output = np.concatenate(eval_output, axis=0)
    eval_uk = np.concatenate(eval_uk, axis=0)
    evaluator = utils.PredictionEvaluator_2(eval_target, args.num_classes)
    for i in range(10):
        t_clss_acc, t_aug_cls_acc = evaluator.evaluate(eval_output, eval_uk, i*0.1)
        print("epslion {:.2f}, mean_aug_class_acc {}, aug_cls_acc {}".format(i*0.1, t_clss_acc, t_aug_cls_acc))


    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    model.train(mode=True)

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()
