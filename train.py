"""Train per-sample"""

import os
import argparse
import numpy as np
import scipy.io as sio
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

from algs import *
from config import *
from utils import *


def main():
  parser = argparse.ArgumentParser()

  # Basic settings
  parser.add_argument('--dataset', type=str, required=True,
                      choices=['celeba', 'cifar10', 'cifar100', 'compas'])
  parser.add_argument('--data_root', type=str)
  parser.add_argument('--device', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--save_file', type=str)
  parser.add_argument('--load_file', type=str)
  parser.add_argument('--load_warmup', type=str)
  parser.add_argument('--n_val', type=int, help='Size of the validation set')
  parser.add_argument('--download', default=False, action='store_true')
  parser.add_argument('--verbose', default=False, action='store_true')

  # Training settings
  parser.add_argument('--alg', type=str, help='Training algorithm', required=True,
                      choices=['uniform', 'adalpboost', 'lpboost'])
  parser.add_argument('--width', type=int, help='Width of Wide ResNet')
  parser.add_argument('--epochs', type=int, help='Number of training epochs')
  parser.add_argument('--iters_per_epoch', default=100, type=int)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--lr', type=float)
  parser.add_argument('--wd', type=float)
  parser.add_argument('--scheduler', type=str)
  parser.add_argument('--alpha', default=0.01, type=float)
  parser.add_argument('--beta', type=float)
  parser.add_argument('--eta', default=1.0, type=float)
  parser.add_argument('--warmup', default=0, type=int)
  parser.add_argument('--num_workers', type=int)
  parser.add_argument('--pin_memory', action='store_true')


  args = parser.parse_args()
  populate_config(args.dataset, args)
  print('Dataset: {}'.format(args.dataset))
  print('Validation set size: {}'.format(args.n_val))
  print('Training algorithm: {}'.format(args.alg))
  print('Width: {}'.format(args.width))
  print('Batch size: {}'.format(args.batch_size))
  print('Epochs: {}'.format(args.epochs))
  print('Iterations per epoch: {}'.format(args.iters_per_epoch))
  print('Warmup epochs: {}'.format(args.warmup))
  print('lr: {}'.format(args.lr))
  print('wd: {}'.format(args.wd))
  print('alpha: {}'.format(args.alpha))
  print('beta: {}'.format(args.beta))
  print('eta: {}'.format(args.eta))

  device = args.device
  if args.save_file is not None:
    d = os.path.dirname(os.path.abspath(args.save_file))
    if not os.path.isdir(d):
      os.makedirs(d)

  # Prepare dataset
  dataset_train, dataset_test_train, dataset_valid, \
  dataset_test, model, label_id = get_dataset(args)
  # Build model
  model = model.to(device)

  # Fix seed for reproducibility
  if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
  else:
    cudnn.benchmark = True

  trainloader = DataLoader(dataset_train, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers,
                           pin_memory=args.pin_memory)
  test_trainloader = DataLoader(dataset_test_train, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory)  # only for test
  testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)
  validloader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=args.pin_memory)

  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.wd)
  criterion = get_criterion(args.dataset)
  scheduler = None
  if args.scheduler is not None:
    milestones = args.scheduler.split(',')
    milestones = [int(s) for s in milestones]
    print('scheduler: {}'.format(milestones))
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
  print('alpha: {}'.format(args.alpha))
  print('eta: {}'.format(args.eta))
  print('Num workers: {}'.format(args.num_workers))
  print('Pin memory: {}'.format(args.pin_memory))
  print('Seed: {}'.format(args.seed))

  # Training
  # 1. Warm up with ERM
  warmup = False
  if args.load_warmup is None:
    for epoch in range(args.warmup):
      print('=== Warmup (epoch={}) ==='.format(epoch + 1))
      timed_run(erm, model, trainloader, optimizer,
                criterion, None, device, 0)
      timed_run(test, model, testloader, criterion, device, label_id)
      warmup = True
    warmup_state = {
      'model': deepcopy(model.state_dict()),
      'optimizer': deepcopy(optimizer.state_dict()),
    }
  else:
    warmup = True
    print('==> Loading warmup state from {}...'.format(args.load_warmup))
    with open(args.load_warmup, 'rb') as f:
      warmup_state = torch.load(f)
    model.load_state_dict(warmup_state['model'])
    optimizer.load_state_dict(warmup_state['optimizer'])

  # 2. Boosting
  val_avg_acc = []
  val_avg_loss = []
  val_correct = []
  val_loss = []
  test_avg_acc = []
  test_avg_loss = []
  test_correct = []
  test_loss = []
  train_avg_acc = []
  train_avg_loss = []
  train_correct = []
  train_loss = []
  train_sample_weights = []
  obj_value = []

  start_epoch = 0
  sample_accuracy_history = np.zeros((0, len(dataset_train)), dtype=np.float)
  sample_losses_history = np.zeros((0, len(dataset_train)), dtype=np.float)
  sample_weights_history = np.zeros((0, len(dataset_train)), dtype=np.float)
  if args.load_file is not None:
    # Load mat file
    print('==> Loading training history from {}...'.format(args.load_file))
    mat = sio.loadmat(args.load_file)
    num_epochs = mat['test_avg_acc'].shape[1]
    test_avg_acc = [mat['test_avg_acc'][0,i] for i in range(num_epochs)]
    test_avg_loss = [mat['test_avg_loss'][0,i] for i in range(num_epochs)]
    test_correct = [mat['test_correct'][i,:] for i in range(num_epochs)]
    test_loss = [mat['test_loss'][i,:] for i in range(num_epochs)]
    val_avg_acc = [mat['val_avg_acc'][0,i] for i in range(num_epochs)]
    val_avg_loss = [mat['val_avg_loss'][0,i] for i in range(num_epochs)]
    val_correct = [mat['val_correct'][i,:] for i in range(num_epochs)]
    val_loss = [mat['val_loss'][i,:] for i in range(num_epochs)]
    train_avg_acc = [mat['train_avg_acc'][0,i] for i in range(num_epochs)]
    train_avg_loss = [mat['train_avg_loss'][0,i] for i in range(num_epochs)]
    train_correct = [mat['train_correct'][i,:] for i in range(num_epochs)]
    train_loss = [mat['train_loss'][i,:] for i in range(num_epochs)]
    train_sample_weights = [mat['train_sample_weights'][i,:] for i in range(num_epochs)]
    sample_accuracy_history = mat['train_correct']
    sample_losses_history = mat['train_loss']
    sample_weights_history = mat['train_sample_weights']
    start_epoch = num_epochs if warmup else num_epochs - 1
    print('Starting from epoch {}.'.format(start_epoch + 1))

  def test_epoch(epoch):
    print('=== Validation (epoch={}) ==='.format(epoch))
    a, b, c, d = timed_run(test, model, validloader, criterion, device, label_id)
    val_avg_acc.append(a)
    val_avg_loss.append(b)
    val_correct.append(c)
    val_loss.append(d)
    print('=== Test (epoch={}) ==='.format(epoch))
    a, b, c, d = timed_run(test, model, testloader, criterion, device, label_id)
    test_avg_acc.append(a)
    test_avg_loss.append(b)
    test_correct.append(c)
    test_loss.append(d)
    print('=== Test over Train set (epoch={}) ==='.format(epoch))
    a, b, c, d = timed_run(test, model, test_trainloader, criterion, device, label_id)
    train_avg_acc.append(a)
    train_avg_loss.append(b)
    train_correct.append(c)
    train_loss.append(d)
    return c, d

  if warmup and args.load_file is None:
    print('==> Testing warmup model...')
    test_epoch(0)

  min_weighted_avg_acc = 1.0
  for epoch in range(start_epoch, args.epochs):
    print('===Train(epoch={})==='.format(epoch + 1))
    sample_weights = get_sample_weights(args, sample_accuracy_history,
                                        sample_losses_history, sample_weights_history,
                                        obj_value, np.array(val_correct))
    print('Weight max: {}'.format(sample_weights.max()))
    train_sample_weights.append(sample_weights)
    sample_weights_history = np.concatenate((sample_weights_history,
                                             sample_weights.reshape(1, len(dataset_train))))

    sweight = torch.tensor(sample_weights).to(device)
    sampler = WeightedRandomSampler(sweight, args.iters_per_epoch * args.batch_size,
                                    replacement=True)
    trainloader = DataLoader(dataset_train, batch_size=args.batch_size,
                             sampler=sampler, num_workers=args.num_workers,
                             pin_memory=args.pin_memory)
    timed_run(erm, model, trainloader, optimizer, criterion,
              scheduler, device, args.iters_per_epoch)

    sample_accuracy, sample_losses = test_epoch(epoch + 1)
    sample_accuracy_history = np.concatenate(
      (sample_accuracy_history, sample_accuracy.reshape(1, len(dataset_train))))
    sample_losses_history = np.concatenate(
      (sample_losses_history, sample_losses.reshape(1, len(dataset_train))))
    weighted_avg_acc = sample_weights @ sample_accuracy
    print('Weighted average accuracy: {}'.format(weighted_avg_acc))
    min_weighted_avg_acc = min(min_weighted_avg_acc, weighted_avg_acc)

    model.load_state_dict(warmup_state['model'])
    optimizer.load_state_dict(warmup_state['optimizer'])
    if scheduler is not None:
      scheduler.last_epoch = 0

  # Final result
  print('==> Training complete.')
  print('Min Weighted Average Acc: {}'.format(min_weighted_avg_acc))

  # Save the results
  if args.save_file is not None:
    print('==>Saving training history to {}...'.format(args.save_file))
    mat = {
      'test_avg_acc': np.array(test_avg_acc),
      'test_avg_loss': np.array(test_avg_loss),
      'test_correct': np.array(test_correct),
      'test_loss': np.array(test_loss),
      'val_avg_acc': np.array(val_avg_acc),
      'val_avg_loss': np.array(val_avg_loss),
      'val_correct': np.array(val_correct),
      'val_loss': np.array(val_loss),
      'train_avg_acc': np.array(train_avg_acc),
      'train_avg_loss': np.array(train_avg_loss),
      'train_correct': np.array(train_correct),
      'train_loss': np.array(train_loss),
      'train_sample_weights': np.array(train_sample_weights),
    }
    if len(obj_value) > 0:
      mat['obj_value'] = np.array(obj_value)
    sio.savemat(args.save_file, mat)

  print('Done.')


def get_sample_weights(args, sample_accuracy_history,
                       sample_losses_history, sample_weights_history,
                       obj_value, val_history=None):
  # Input shape: each history - (epoch, num_samples)
  # Output: group_weights  size: (num_samples,)
  if len(sample_accuracy_history) == 0:
    return uniform(sample_losses_history)

  if args.alg == 'uniform':
    return uniform(sample_losses_history)
  elif args.alg == 'adalpboost':
    return adalpboost(sample_accuracy_history, sample_weights_history, args.eta)
  elif args.alg == 'lpboost':
    return lpboost(sample_accuracy_history, obj_value, args.alpha, args.beta, args.verbose)
  else:
    raise NotImplementedError


if __name__ == '__main__':
  main()
