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
        image_size = 256
        input_size = 224
        student_model = models.__dict__[args.arch](num_classes = args.num_classes,
                                                   pretrained = args.pretrained,
                                                   )
        teacher_model = models.__dict__[args.arch](num_classes=args.num_classes,
                                                   pretrained=args.pretrained,)



    student_model.cuda()
    student_params = list(student_model.parameters())


    teacher_model.cuda()
    teacher_params = list(teacher_model.parameters())


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
    weight = weight[:args.num_classes]
    if not args.use_weight:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    ignored_params = list(map(id, student_model.from_scratch.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad==True,
                         student_model.parameters())
    if args.pretrained==True:
        student_optimizer = torch.optim.Adam([
                                    {'params': base_params},
                                    {'params': student_model.from_scratch.parameters(), 'lr': args.base_lr}
                                    ] ,args.base_lr*0.1)
    else:
        student_optimizer = torch.optim.Adam(student_model.parameters(),
                                     args.base_lr)
    # optionally resume from a checkpoint
    teacher_optimizer = utils.WeightEMA(teacher_params, student_params,
                                        alpha=args.teacher_alpha)

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
           student_optimizer=student_optimizer,
           lr_scheduler=lr_scheduler,
           start_iter=last_iter+1, tb_logger = tb_logger,
          teacher_model=teacher_model,
          teacher_optimizer=teacher_optimizer)

def train(train_source_loader, train_target_loader, val_loader,
          student_model, criterion, student_optimizer,
           lr_scheduler, start_iter, tb_logger,
          teacher_model=None,
          teacher_optimizer=None):

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

        input_target, input_target1, label_target = batch_target
        curr_step = start_iter + i

        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]
        # measure data loading time
        data_time.update(time.time() - end)


        label_source = Variable(label_source).cuda(async=True)
        input_source = Variable(input_source).cuda()

        input_target = Variable(input_target).cuda(async=True)
        input_target1 = Variable(input_target1).cuda(async=True)
        # compute output for source data
        source_output, source_output2 = student_model(input_source)

        # measure accuracy and record loss
        softmax_source_output = F.softmax(source_output, dim=1)

        #loss for known class
        known_ind = label_source != args.num_classes
        if args.double_softmax:
            loss_cls = criterion(softmax_source_output[known_ind], label_source[known_ind])
        else:
            loss_cls = criterion(source_output[known_ind], label_source[known_ind])


        loss = loss_cls
        uk_label = label_source.clone()
        uk_label[uk_label!=args.num_classes]=0
        uk_label[uk_label==args.num_classes]=1
        uk_label = uk_label.float().unsqueeze(1)
        loss_uk = criterion_uk(source_output2, uk_label)


        loss_entropy = torch.mean(
            torch.mul(softmax_source_output[label_source==args.num_classes],
                    torch.log(softmax_source_output[label_source==args.num_classes])))
        loss += args.lambda_uk * loss_uk
        loss += args.lambda_entropy * loss_entropy
        #loss for unknown class
        #integrate loss_cls and loss_entropy
        #compute accuracy




        # for target data

        stu_out, stu_out2 = student_model(input_target)
        tea_out, tea_out2 = teacher_model(input_target1)

        loss_aug, conf_mask, loss_cls_bal = \
            utils.new_compute_aug_loss_enp(stu_out, tea_out, args.aug_thresh,
                                           tea_out2,
                                           args.cls_balance, args)
        conf_mask_count = torch.sum(conf_mask) / args.batch_size
        loss_aug = torch.mean(loss_aug)
        loss += args.lambda_aug * loss_aug
        loss += args.cls_balance * args.lambda_aug * loss_cls_bal

        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()


        eval_output.append(softmax_source_output.cpu().data.numpy())
        eval_target.append(label_source.cpu().data.numpy())
        eval_uk.append(source_output2.cpu().data.numpy())
        prec1, prec5 = accuracy_2(softmax_source_output.data, source_output2.data, label_source, 0.5, topk=(1, 5))




        losses.update(loss_cls.item())
        top1.update(prec1.item())
        top5.update(prec5.item())
        # compute gradient and do SGD step
        losses_cls_uk.update(loss_uk.item())
        losses_entropy.update(loss_entropy.item())
        losses_aug.update(loss_aug.item())
        losses_bal.update(loss_cls.item())


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
                        'Time: {batch_time.avg:.3f}\t'
                        'Data: {data_time.avg:.3f}\t'
                        'loss: {loss.avg:.4f}\t'
                        'loss_uk: {loss_uk.avg:.4f}\t'
                        'loss_aug: {loss_aug.avg:.4f}\t'
                        'loss_bal: {loss_bal.avg:.4f}\t'
                        'loss_entropy: {loss_entropy.avg:.4f}\t'
                        'Prec@1: {top1.avg:.3f}\t'
                        'Prec@5: {top5.avg:.3f}\t'
                        'lr: {lr:.6f}'.format(
                   curr_step, len(train_source_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   loss_uk=losses_cls_uk,
                   top1=top1, top5=top5,
                   loss_aug=losses_aug,
                   loss_bal=losses_bal,
                   loss_entropy=losses_entropy,
                   lr=current_lr))

        if (curr_step+1)%args.val_freq == 0 :

            eval_target = np.concatenate(eval_target, axis=0)
            eval_output = np.concatenate(eval_output, axis=0)
            eval_uk = np.concatenate(eval_uk, axis=0)
            evaluator = utils.PredictionEvaluator_2(eval_target, args.num_classes)
            mean_aug_class_acc, aug_cls_acc = evaluator.evaluate(eval_output, eval_uk, 0.5)
            eval_target = []
            eval_output = []
            eval_uk = []
            logger.info("mean_cls_acc: {} cls {}".format(mean_aug_class_acc, aug_cls_acc))

            for cls in range(args.num_classes):
                tb_logger.add_scalar('acc_'+args.class_name[cls], aug_cls_acc[cls], curr_step)


            val_loss, prec1, prec5, mean_aug_class_acc, aug_cls_acc, best_acc  = validate(val_loader, teacher_model, criterion)
            if not tb_logger is None:
                tb_logger.add_scalar('loss_val', val_loss, curr_step)
                tb_logger.add_scalar('acc1_val', prec1, curr_step)
                tb_logger.add_scalar('acc5_val', prec5, curr_step)
                tb_logger.add_scalar('best_acc_val', best_acc, curr_step)
            logger.info("evaluate step mean:{} class:{} best_acc {}".format(mean_aug_class_acc, aug_cls_acc, best_acc))


            # remember best prec@1 and save checkpoint
            is_best = best_acc > best_prec1
            best_prec1 = max(best_acc, best_prec1)
            logger.info("best val prec1 {}".format(best_prec1))
            save_checkpoint({
                'step': curr_step,
                'arch': args.arch,
                'state_dict': teacher_model.state_dict(),
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
    eval_target = []
    eval_output = []
    eval_uk = []
    for i, (input, target) in enumerate(val_loader):
        input = Variable(input, volatile=True).cuda()
        target =Variable(target, volatile=True).cuda()

        # compute output
        output, output1 = model(input)

        known_ind = target!=args.num_classes
        # measure accuracy and record loss
        softmax_output = F.softmax(output, dim=1)
        #loss for known class
        loss = criterion(output[known_ind], target[known_ind])

        #losses.update(loss.item())
        eval_target.append(target.cpu().data.numpy())
        eval_output.append(softmax_output.cpu().data.numpy())
        eval_uk.append(output1.cpu().data.numpy())

        prec1, prec5 = accuracy_2(softmax_output.data,
                                  output1.data,
                                  target, 0.5, topk=(1, 5))

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
    eval_target = np.concatenate(eval_target, axis=0)
    eval_output = np.concatenate(eval_output, axis=0)
    eval_uk = np.concatenate(eval_uk, axis=0)
    evaluator = utils.PredictionEvaluator_2(eval_target, args.num_classes)
    mean_aug_class_acc, aug_cls_acc = evaluator.evaluate(eval_output, eval_uk, 0.5)
    best_acc = 0
    for i in range(10):
        t_clss_acc, t_aug_cls_acc = evaluator.evaluate(eval_output, eval_uk, i*0.1)
        best_acc = max(best_acc, t_clss_acc)
        print("epslion {:.2f}, mean_aug_class_acc {}, aug_cls_acc {}".format(i*0.1, t_clss_acc, t_aug_cls_acc))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    model.train(mode=True)


    return losses.avg, top1.avg, top5.avg, mean_aug_class_acc, aug_cls_acc, best_acc*100

if __name__ == '__main__':
    main()
