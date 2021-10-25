
from torchvision.datasets import CIFAR10, CIFAR100, CelebA
import pandas as pd
import scipy.io as sio
import numpy as np

from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.models import resnet18
from wrn import wrn_28
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


default_config = {
  'celeba': {
    'device': 'cuda',
    'epochs': 30,
    'iters_per_epoch': 100,
    'batch_size': 400,
    'lr': 0.001,
    'wd': 0.001,
    'num_workers': 4,
    'pin_memory': True,
  },
  'cifar10': {
    'device': 'cuda',
    'width': 1,
    'epochs': 30,
    'iters_per_epoch': 100,
    'batch_size': 128,
    'lr': 0.1,
    'wd': 5e-4,
    'num_workers': 4,
    'pin_memory': True,
    'n_val': 5000,
  },
  'cifar100': {
    'device': 'cuda',
    'width': 10,
    'epochs': 30,
    'iters_per_epoch': 100,
    'batch_size': 128,
    'lr': 0.01,
    'wd': 5e-4,
    'num_workers': 4,
    'pin_memory': True,
    'n_val': 5000,
  },
  'compas': {
    'device': 'cpu',
    'epochs': 100,
    'iters_per_epoch': 1000,
    'batch_size': 128,
    'lr': 0.01,
    'wd': 0,
    'num_workers': 0,
    'pin_memory': False,
  },
}


def preprocess_compas(df: pd.DataFrame):
  """Preprocess dataset"""

  columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
             'age',
             'c_charge_degree',
             'sex', 'race', 'is_recid']
  target_variable = 'is_recid'

  df = df[['id'] + columns].drop_duplicates()
  df = df[columns]

  race_dict = {'African-American': 1, 'Caucasian': 0}
  df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 2, axis=1).astype(
    'category')

  sex_map = {'Female': 0, 'Male': 1}
  df['sex'] = df['sex'].map(sex_map)

  c_charge_degree_map = {'F': 0, 'M': 1}
  df['c_charge_degree'] = df['c_charge_degree'].map(c_charge_degree_map)

  X = df.drop([target_variable], axis=1)
  y = df[target_variable]
  return X, y


class MyDataset(Dataset):
  def __init__(self, X, y):
    super(MyDataset, self).__init__()
    self.X = X
    self.y = y
    self.attr = X

  def __getitem__(self, item):
    return self.X[item], self.y[item]

  def __len__(self):
    return len(self.X)


def get_transform_celebA(augment, target_w=None, target_h=None):
  # Reference: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py#L80
  orig_w = 178
  orig_h = 218
  orig_min_dim = min(orig_w, orig_h)
  if target_w is not None and target_h is not None:
    target_resolution = (target_w, target_h)
  else:
    target_resolution = (orig_w, orig_h)

  if not augment:
    transform = transforms.Compose([
      transforms.CenterCrop(orig_min_dim),
      transforms.Resize(target_resolution),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  else:
    # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
    transform = transforms.Compose([
      transforms.RandomResizedCrop(
        target_resolution,
        scale=(0.7, 1.0),
        ratio=(1.0, 1.3333333333333333),
        interpolation=InterpolationMode.BILINEAR),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  return transform


def get_dataset(args):
  name = args.dataset
  data_root = args.data_root
  download = args.download
  if name == 'celeba':
    target_w = 224
    target_h = 224
    transform_train = get_transform_celebA(True, target_w, target_h)
    transform_test = get_transform_celebA(False, target_w, target_h)

    dataset_test = CelebA(data_root, split='test', target_type='attr',
                          transform=transform_test, download=download)
    dataset_valid = CelebA(data_root, split='valid', target_type='attr',
                           transform=transform_test, download=False)
    target_idx = 9  # Blond
    dataset_train = CelebA(data_root, split='train', target_type='attr',
                           transform=transform_train,
                           target_transform=lambda t: t[target_idx])
    dataset_test_train = CelebA(data_root, split='train', target_type='attr',
                                transform=transform_test)
    n_classes = 2
    model = resnet18()
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
    label_id = lambda t: t[:, target_idx]

  elif name == 'cifar10' or name == 'cifar100':
    dataset = CIFAR10 if name == 'cifar10' else CIFAR100
    width = args.width
    num_classes = 10 if name == 'cifar10' else 100

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)) if name == 'cifar10' else \
           transforms.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    n_val = args.n_val
    dataset_test = dataset(
        root=data_root, train=False, download=download, transform=transform_test)
    dataset_test.data = np.array(dataset_test.data)
    dataset_test.targets = np.array(dataset_test.targets)
    dataset_train = dataset(
        root=data_root, train=True, transform=transform_train)
    dataset_train.data = np.array(dataset_train.data)[n_val:]
    dataset_train.targets = np.array(dataset_train.targets)[n_val:]
    dataset_test_train = dataset(
        root=data_root, train=True, transform=transform_test)
    dataset_test_train.data = np.array(dataset_test_train.data)[n_val:]
    dataset_test_train.targets = np.array(dataset_test_train.targets)[n_val:]
    dataset_valid = dataset(
        root=data_root, train=True, transform=transform_test)
    dataset_valid.data = np.array(dataset_valid.data)[:n_val]
    dataset_valid.targets = np.array(dataset_valid.targets)[:n_val]
    model = wrn_28(num_classes=num_classes, width=width)
    label_id = None

  elif name == 'compas':
    df = pd.read_csv('compas-scores-two-years.csv')
    X, y = preprocess_compas(df)
    input_dim = len(X.columns)
    X, y = X.to_numpy().astype('float32'), y.to_numpy()
    X[:, 4] /= 10
    X[X[:, 7] > 0, 7] = 1  # Race: White (0) and Others (1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, shuffle=True)

    if args.data_mat is not None:
      # User dataset
      mat = sio.loadmat(args.data_mat)
      X_train = mat['X_train'].astype('float32')
      y_train = mat['y_train'].flatten()

    dataset_train = MyDataset(X_train, y_train)
    dataset_test_train = MyDataset(X_train, y_train)
    dataset_valid = MyDataset(X_test, y_test)
    dataset_test = MyDataset(X_test, y_test)
    model = nn.Sequential(nn.Linear(input_dim, 100, bias=True),
                          nn.ReLU(inplace=True),
                          nn.Linear(100, 100, bias=True),
                          nn.ReLU(inplace=True),
                          nn.Linear(100, 2, bias=True))
    label_id = None

  else:
    raise NotImplementedError

  return dataset_train, dataset_test_train, dataset_valid, dataset_test, model, label_id


def get_criterion(name):
  criterion = nn.CrossEntropyLoss(reduction='none')
  return criterion


def populate_config(name, args):
  for k, v in default_config[name].items():
    if getattr(args, k) is None:
      setattr(args, k, v)
