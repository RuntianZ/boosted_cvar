"""Train per-sample"""

import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from algs import *
from config import *
from utils import *


def to_numpy(x):
  if isinstance(x, np.ndarray):
    return x
  return x.numpy()


def main():
  parser = argparse.ArgumentParser()

  # Basic settings
  parser.add_argument('--dataset', default='celeba', type=str)
  parser.add_argument('--data_root', type=str)
  parser.add_argument('--device', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--save_file', type=str)
  parser.add_argument('--load_file', type=str)
  parser.add_argument('--download', default=False, action='store_true')
  parser.add_argument('--data_mat', type=str)
  parser.add_argument('--verbose', default=False, action='store_true')

  # Training settings
  parser.add_argument('--alg', type=str)
  parser.add_argument('--epochs', type=int)
  parser.add_argument('--iters_per_epoch', default=100, type=int)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--lr', type=float)
  parser.add_argument('--wd', type=float)
  parser.add_argument('--scheduler', type=str)
  parser.add_argument('--alpha', default=0.01, type=float)
  parser.add_argument('--beta', type=float, default=100)
  parser.add_argument('--gamma', type=float, default=0.25)
  parser.add_argument('--eps', type=float)
  parser.add_argument('--warmup', default=0, type=int)
  parser.add_argument('--retrain', default=False, action='store_true')
  parser.add_argument('--num_workers', type=int)
  parser.add_argument('--pin_memory', default=False, action='store_true')
  parser.add_argument('--dro_step_size', default=1.0, type=float)
  parser.add_argument('--primal_constraint', '-c', default=0, type=float)
  parser.add_argument('--n_val', type=int)

  args = parser.parse_args()
  if not args.pin_memory:
    args.pin_memory = None
  populate_config(args.dataset, args)
  print('Algorithm: {}'.format(args.alg))
  print('Dataset: {}'.format(args.dataset))
  print('Batch size: {}'.format(args.batch_size))
  print('Epochs: {}'.format(args.epochs))
  print('Iters per epoch: {}'.format(args.iters_per_epoch))
  print('Warmup epochs: {}'.format(args.warmup))
  print('Retrain: {}'.format(args.retrain))
  print('lr: {}'.format(args.lr))
  print('wd: {}'.format(args.wd))
  print('alpha: {}'.format(args.alpha))
  print('beta: {}'.format(args.beta))
  print('gamma: {}'.format(args.gamma))
  print('eps: {}'.format(args.eps))
  print('Num workers: {}'.format(args.num_workers))
  print('DRO step size: {}'.format(args.dro_step_size))
  print('primal_constraint: {}'.format(args.primal_constraint))
  print('Seed: {}'.format(args.seed))
  print('n_val: {}'.format(args.n_val))

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

  # Training
  # 1. Warm up with ERM
  for epoch in range(args.warmup):
    # Warmup is still needed even if loaded from mat file
    print('===Warmup(epoch={})==='.format(epoch + 1))
    timed_run(erm, model, trainloader, optimizer,
              criterion, None, device, 0)
    timed_run(test, model, testloader, criterion, device, label_id)
    if scheduler is not None:
      scheduler.step()

  warmup_state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }

  with open(args.save_file, 'wb') as f:
    torch.save(warmup_state, f)

if __name__ == '__main__':
  main()