import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from algs import find_opt_lbd


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', type=str)
  args = parser.parse_args()

  mat = sio.loadmat(args.file)
  val_correct = mat['val_correct']
  train_correct = mat['train_correct']
  test_correct = mat['test_correct']

  erm_acc = mat['test_avg_acc'].max()
  tmean = test_correct.mean(0)
  trmean = train_correct.mean(0)
  tmean.sort()
  trmean.sort()
  m_arr = []
  a_arr = []
  avg_arr = []
  tm_arr = []
  avgm_arr = []
  erm_arr = []
  tr_arr = []
  trm_arr = []

  for k in range(1, 51):
    alpha = 0.01 * k
    print(alpha)
    m_arr.append(alpha)

    lbd = find_opt_lbd(val_correct, alpha)
    te = lbd @ test_correct
    te.sort()

    nt = len(te)
    mt = int(alpha * nt)
    a = te[:mt].mean()
    a_arr.append(a)
    avg = te.mean()
    avg_arr.append(avg)
    tm = tmean[:mt].mean()
    tm_arr.append(tm)
    avgm = tmean.mean()
    avgm_arr.append(avgm)
    if alpha < 1 - erm_acc:
      e = 0
    else:
      e = 1 - (1 - erm_acc) / alpha
    erm_arr.append(e)

    tr = lbd @ train_correct
    tr.sort()
    ntr = len(tr)
    mtr = int(alpha * ntr)
    tr = tr[:mtr].mean()
    tr_arr.append(tr)
    trm = trmean[:mtr].mean()
    trm_arr.append(trm)

  m_arr = np.array(m_arr)
  erm_arr = np.array(erm_arr)
  tm_arr = np.array(tm_arr)
  a_arr = np.array(a_arr)

  plt.rcParams.update({'font.size': 22})
  plt.figure(figsize=(8, 6), dpi=80)

  plt.plot(m_arr, 1 - erm_arr, label='ERM', linewidth=3.0)
  plt.plot(m_arr, 1 - tm_arr, label='AdaBoost + Average', linewidth=3.0)
  plt.plot(m_arr, 1 - a_arr, label='AdaLPBoost', linewidth=3.0,
           alpha=0.5, marker='^', markersize=7)

  plt.legend()
  plt.xlim(0, max(m_arr) + 0.005)
  plt.xlabel(r'$\alpha$')
  plt.ylabel(r'$\alpha$-CVaR Zero-one Loss')
  plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
  plt.show()


if __name__ == '__main__':
  main()