import torch
from torch import nn


net  = nn.Sequential(nn.Conv1d(20, 10, 4), nn.Conv1d(10, 5, 3))

print(net[1])