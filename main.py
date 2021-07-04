import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Learn N-point DFT matrix.")
parser.add_argument("--N",  help="Length of DFT.", nargs="?", default=32, type=int)
parser.add_argument("--lr",  help="Learning rate. Helps to increase as N increases.", nargs="?", default=0.01, type=int)
parser.add_argument("--batch", help="Batch size. Increasing doesn't seem to change much, but feel free to play around wiht it.", nargs="?", default=1, type=int)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

SIG_LENGTH = args.N
BATCH = args.batch
LR = args.lr
LOSS_TOL = 0.00015
VERBOSE = args.verbose

W_FFT = torch.zeros(SIG_LENGTH, SIG_LENGTH, dtype=torch.complex64)
for n in range(SIG_LENGTH):
    x = -n * torch.arange(SIG_LENGTH) * 2*np.pi/SIG_LENGTH
    cosine = torch.cos(x).reshape(-1, 1)
    sine = torch.sin(x).reshape(-1, 1)
    W_FFT[n, :] = torch.view_as_complex(torch.cat((cosine, sine), dim=1))

def ifft_loss(output, target):
    y = torch.fft.ifft(output)
    error = y - target
    return torch.mean(error.abs())

class Model(nn.Module):
    def __init__(self, N):
        super(Model, self).__init__()
        self.linear = nn.Linear(N, N, bias=False, dtype=torch.complex64)

    def forward(self, x):
        return self.linear(x)

model = Model(SIG_LENGTH)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

loss_sv = []

iters = 0
a = 0
while True:
    iters += 1
    x = torch.rand(BATCH, SIG_LENGTH, dtype=torch.complex64) - torch.view_as_complex(torch.Tensor([0.5, 0.5]))
    pred = model(x)
    loss = ifft_loss(pred, x)
    loss.backward()
    loss_sv.append(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    if(VERBOSE and iters % 2500 == 0):
        print(f"Iteration {iters}, loss = {loss_sv[-1]}")
    if(loss.item() < LOSS_TOL):
        break

print(f"Loss = {loss_sv[-1]}, took {iters} iterations")
plt.plot([x for x in range(len(loss_sv))], loss_sv)
plt.show()

W = list(model.parameters())[0].detach()

error = torch.abs(torch.mean(torch.square(W - W_FFT)))
print(f"Weights error: {error}")

fig, axs = plt.subplots(2, 2)
axs[0, 0].matshow(torch.real(W_FFT), cmap="Greys")
axs[0, 0].set_title("Actual Weights (Real)")
axs[0, 1].matshow(torch.imag(W_FFT), cmap="Greys")
axs[0, 1].set_title("Actual Weights (Imaginary)")
axs[1, 0].matshow(torch.real(W), cmap="Greys")
axs[1, 0].set_title("Learned Weights (Real)")
axs[1, 1].matshow(torch.imag(W), cmap="Greys")
axs[1, 1].set_title("Learned Weights (Imaginary)")
plt.show()