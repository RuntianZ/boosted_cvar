import numpy as np
import cvxpy as cp
import mosek

import torch
import torch.nn
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F


###########################################
# Algorithms for selecting sample weights
def uniform(sample_losses_history):
  num_samples = sample_losses_history.shape[1]
  a = np.ones((num_samples,), dtype=np.float) / num_samples
  return a


def adalpboost(sample_accuracy_history, sample_weights_history, eta):
  acc = sample_accuracy_history[-1]
  weight = sample_weights_history[-1]
  weight *= np.exp(-eta * acc)
  weight /= weight.sum()
  return weight


# (Regularized) LPBoost
# Dual problem of LPBoost
def lpboost(sample_accuracy_history, obj_value, alpha,
            beta=None, verbose=False, solve_for_lbd=False):
  # Input shape: each history - (epoch, num_samples)
  # Output: group_weights  size: (num_samples,)
  num_epochs, n = sample_accuracy_history.shape
  m = alpha * n

  w = cp.Variable(n)
  g = cp.Variable()
  objective = cp.Minimize(g) if beta is None else cp.Minimize(g - cp.sum(cp.entr(w)) / beta)
  constraints = [sample_accuracy_history[i, :] @ w <= g for i in range(num_epochs)]
  constraints.append(cp.sum(w) == 1)
  constraints.append(0 <= w)
  constraints.append(w <= 1 / m)

  prob = cp.Problem(objective, constraints)
  result = lpsolver(prob, verbose)
  if obj_value is not None:
    obj_value.append(result)

  if solve_for_lbd:
    lbd = [constraints[i].dual_value for i in range(num_epochs)]
    lbd = np.array(lbd)
    lbd[lbd < 0] = 0
    lbd /= lbd.sum()
    return lbd

  ans = w.value
  ans[ans < 0] = 0
  ans /= ans.sum()
  return ans


############################
# Find optimal lambda (model weights)
# Solve the dual problem of LPBoost and use the values of primal variables
def find_opt_lbd(acc_history, alpha, verbose=False):
  print('==> Computing optimal model weights...')
  lbd = lpboost(acc_history, None, alpha, None, verbose, True)
  print('Optimal model weights found.')
  print('Model weights: {}'.format(lbd))
  return lbd


###########################################
# Other functions
def test(model: Module, loader: DataLoader, criterion, device: str, label_id):
  """Test the avg and group acc of the model"""

  model.eval()
  total_correct = 0
  total_loss = 0
  total_num = 0
  l_rec = []
  c_rec = []

  with torch.no_grad():
    for _, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)
      labels = targets if label_id is None else label_id(targets)
      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)
      c = (predictions == labels)
      c_rec.append(c.detach().cpu().numpy())
      correct = c.sum().item()
      l = criterion(outputs, labels).view(-1)
      l_rec.append(l.detach().cpu().numpy())
      loss = l.sum().item()
      total_correct += correct
      total_loss += loss
      total_num += len(inputs)

  print('Acc: {} ({} of {})'.format(total_correct / total_num, total_correct, total_num))
  print('Avg Loss: {}'.format(total_loss / total_num))

  l_vec = np.concatenate(l_rec)
  c_vec = np.concatenate(c_rec)

  return total_correct / total_num, total_loss / total_num, \
         c_vec, l_vec


def erm(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
        criterion, scheduler, device: str, iters=0):
  """Empirical Risk Minimization (ERM)"""

  model.train()
  iteri = 0
  for _, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    iteri += 1
    if iteri == iters:
      break
    if scheduler is not None:
      scheduler.step()

############################
# LP Solver
# Try multiple solvers because single solver might fail
def lpsolver(prob, verbose=False):
  print('=== LP Solver ===')
  solvers = [cp.MOSEK, cp.ECOS_BB]
  for s in solvers:
    print('==> Invoking {}...'.format(s))
    try:
      result = prob.solve(solver=s, verbose=verbose)
      return result
    except cp.error.SolverError as e:
      print('==> Solver Error')

  print('==> Invoking MOSEK simplex method...')
  try:
    result = prob.solve(solver=cp.MOSEK,
                      mosek_params={mosek.iparam.optimizer: mosek.optimizertype.free_simplex},
                      bfs=True, verbose=verbose)
    return result
  except cp.error.SolverError as e:
    print('==> Solver Error')

  raise cp.error.SolverError('All solvers failed.')

