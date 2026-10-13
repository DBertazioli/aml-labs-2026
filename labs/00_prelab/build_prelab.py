"""Builds prelab_pytorch.ipynb (student) and prelab_pytorch_solutions.ipynb.
Run: python build_prelab.py
Exercises: ex(student_code, solution_code). Everything else is shared."""
import nbformat as nbf

BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/DBertazioli/aml-labs-2026/blob/main/labs/00_prelab/prelab_pytorch.ipynb)")

C = []  # list of (kind, student_src, solution_src)
def md(s): C.append(("md", s.strip(), s.strip()))
def code(s): C.append(("code", s.strip(), s.strip()))
def ex(student, solution): C.append(("code", student.strip(), solution.strip()))

md(BADGE + """

# Pre-lab: PyTorch warm-up + connection test

**Advanced Machine Learning, UNIMIB.** Do this before the first lab (27 Oct).

- Sections 1-7 (~45 min) are **optional self-study**. Skip what you already know. Some cells are exercises: replace `None` / `# TODO` and run the check cell below it (it uses `assert`; no error means correct). The solutions notebook is `prelab_pytorch_solutions.ipynb`.
- **Section 8 is REQUIRED (~3 min)**: it checks that Colab works for you and that you can reach the class server. Do it at least one day before the lab.

Run all cells from top to bottom (Runtime > Run all works as well, but exercises will stop at the first unsolved TODO).""")

md("""## 1. Setup

Colab: **Runtime > Change runtime type**. The default **CPU** is fine for both labs. A GPU is optional and not needed for anything here.

The cell below is the **only configuration cell**. Do not change it.""")
code('''# ---- CONFIG (do not edit) ----
CLIENT_URL = "https://raw.githubusercontent.com/DBertazioli/aml-labs-2026/main/client/aml_client.py"
# ------------------------------''')
code('''import os, time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| numpy", np.__version__, "| device:", device)
torch.manual_seed(0)''')

md("""## 2. Tensors

A tensor is an n-dimensional array (like numpy) that can live on a GPU and track gradients. Its three key properties: `shape`, `dtype`, `device`.""")
code('''a = torch.zeros(2, 3)              # shape (2, 3), dtype float32
b = torch.arange(6).reshape(2, 3)   # dtype int64
c = torch.randn(2, 3)              # standard normal
print(a.shape, a.dtype, a.device)
print(b.dtype, "->", b.float().dtype)   # convert dtype
print(torch.ones(2, 3, device=device).device)  # create directly on device''')
md("""**Broadcasting**: when shapes differ, size-1 (or missing) dimensions are stretched. `(3, 4) - (4,)` subtracts the vector from every row; `(3, 4) - (3, 1)` subtracts one number per row. Use `keepdim=True` to keep the reduced dimension as size 1.""")
code('''x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(x - x.mean(dim=0))                 # (3,4) - (4,)
print(x - x.mean(dim=1, keepdim=True))   # (3,4) - (3,1)''')
md("**Exercise 2.1** Create a float32 tensor `t` of shape `(3, 4)` containing `0, 1, ..., 11` (row by row).")
ex('''t = None  # TODO''',
   '''t = torch.arange(12, dtype=torch.float32).reshape(3, 4)''')
code('''assert t.shape == (3, 4) and t.dtype == torch.float32 and t[2, 3] == 11
print("ok")''')
md("**Exercise 2.2** Standardize the columns of `t`: subtract each column's mean and divide by its standard deviation (use `std(dim=0)`). Use broadcasting, no loops.")
ex('''z = None  # TODO''',
   '''z = (t - t.mean(dim=0)) / t.std(dim=0)''')
code('''assert z.shape == (3, 4)
assert torch.allclose(z.mean(dim=0), torch.zeros(4), atol=1e-6)
assert torch.allclose(z.std(dim=0), torch.ones(4), atol=1e-5)
print("ok")''')
md("""**numpy interop**: `torch.from_numpy(arr)` and `tensor.numpy()` (CPU only) **share memory** with the original: changing one changes the other. Use `.clone()` / `.copy()` if you want an independent copy. A tensor that requires grad or lives on GPU needs `.detach().cpu().numpy()`.""")
md("**Exercise 2.3** Make `arr` a numpy array `[1., 2., 3.]`, create `s = torch.from_numpy(arr)`, then double `s` **in place** (`s *= 2`) and check what happened to `arr`.")
ex('''arr = None  # TODO: numpy array [1., 2., 3.]
s = None    # TODO: torch tensor sharing memory with arr
# TODO: double s in place''',
   '''arr = np.array([1., 2., 3.])
s = torch.from_numpy(arr)
s *= 2''')
code('''assert np.allclose(arr, [2., 4., 6.]), "arr should have changed too (shared memory)"
print("ok")''')

md("""## 3. Autograd

If a tensor has `requires_grad=True`, PyTorch records the operations on it. Calling `.backward()` on a scalar fills `.grad` of every leaf tensor with the derivative of that scalar.""")
code('''x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x          # dy/dx = 2x + 3 = 7 at x = 2
y.backward()
print(x.grad)''')
md("**Exercise 3.1** For `f(x) = x**3 + 2*x` at `x = 2`, compute the gradient with autograd (`g_auto`) and by hand (`g_true`, derivative `3x^2 + 2`).")
ex('''x = torch.tensor(2.0, requires_grad=True)
f = None  # TODO: f(x)
# TODO: call backward
g_auto = None  # TODO: x.grad
g_true = None  # TODO: analytic derivative at x=2 (a float)''',
   '''x = torch.tensor(2.0, requires_grad=True)
f = x**3 + 2 * x
f.backward()
g_auto = x.grad
g_true = 3 * 2.0**2 + 2''')
code('''assert abs(float(g_auto) - g_true) < 1e-6 and abs(g_true - 14) < 1e-6
print("ok")''')
md("""**Pitfall: gradients accumulate.** Each `.backward()` *adds* to `.grad`. In a training loop you must reset the gradients every step (`optimizer.zero_grad()`), otherwise the step uses the sum of all past gradients.""")
code('''w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = (3 * w)          # d loss / d w = 3 every time
    loss.backward()
    print(f"step {i}: w.grad = {w.grad.item()}")   # 3, 6, 9: wrong without zero_grad''')
md("**Exercise 3.2** Redo the loop above but reset the gradient each step so that `w.grad` is always 3. (For a single tensor use `w.grad.zero_()`; for a model/optimizer use `optimizer.zero_grad()`.)")
ex('''w = torch.tensor(1.0, requires_grad=True)
seen = []
for i in range(3):
    # TODO: reset the gradient here
    loss = 3 * w
    loss.backward()
    seen.append(w.grad.item())''',
   '''w = torch.tensor(1.0, requires_grad=True)
seen = []
for i in range(3):
    if w.grad is not None:
        w.grad.zero_()
    loss = 3 * w
    loss.backward()
    seen.append(w.grad.item())''')
code('''assert seen == [3.0, 3.0, 3.0]
print("ok")''')

md("""## 4. `nn.Module`: an MLP

A model is a class with layers defined in `__init__` and the computation in `forward`. `nn.Linear(i, o)` holds a weight `(o, i)` and a bias `(o,)`. PyTorch's default init for `nn.Linear` is uniform in `[-1/sqrt(i), 1/sqrt(i)]` (Kaiming-uniform with `a=sqrt(5)`), and biases get the same range.

**Output of the model = logits** (raw scores, no softmax).""")
md("**Exercise 4.1** Complete the MLP: 784 -> `hidden` -> `hidden` ... -> 10 with ReLU between layers (no activation after the last layer).")
ex('''class MLP(nn.Module):
    def __init__(self, hidden=128, p_drop=0.0):
        super().__init__()
        self.net = None  # TODO: nn.Sequential(Linear(784,hidden), ReLU, Dropout(p_drop), Linear(hidden,10))
    def forward(self, x):
        return self.net(x)''',
   '''class MLP(nn.Module):
    def __init__(self, hidden=128, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(p_drop),
                                 nn.Linear(hidden, 10))
    def forward(self, x):
        return self.net(x)''')
code('''m = MLP(hidden=128)
assert m(torch.randn(5, 784)).shape == (5, 10)
print(m)''')
md("**Exercise 4.2** Count the parameters, first with a one-liner over `m.parameters()`, then by hand (weights + biases of each layer).")
ex('''n_auto = None  # TODO: sum of p.numel()
n_hand = None  # TODO: (784*128 + 128) + (128*10 + 10)''',
   '''n_auto = sum(p.numel() for p in m.parameters())
n_hand = (784 * 128 + 128) + (128 * 10 + 10)''')
code('''assert n_auto == n_hand == 101770
w = m.net[0].weight
print("first layer weight std:", w.std().item(), "| bound 1/sqrt(784) =", 784**-0.5)''')

md("""## 5. Training loop on FashionMNIST

FashionMNIST: 60,000 train + 10,000 test grayscale images of 28x28 pixels, 10 clothing classes. It is small enough to keep **entirely as tensors on the device** and do **manual mini-batching** (random permutation, slice). This is much faster than a `DataLoader` here. (`torch.utils.data.DataLoader` exists and you will want it for bigger data or augmentations.)

The first run downloads ~30 MB.""")
code('''from torchvision import datasets
root = os.environ.get("AML_DATA", "data")          # Colab: just "data"
tr = datasets.FashionMNIST(root, train=True, download=True)
te = datasets.FashionMNIST(root, train=False, download=True)

def prep(d):  # uint8 (N,28,28) -> float (N,784), standardized
    return ((d.data.float() / 255 - 0.286) / 0.353).view(len(d), -1).to(device), d.targets.to(device)
Xtr, ytr = prep(tr); Xte, yte = prep(te)
print(Xtr.shape, ytr.shape, Xte.shape, Xtr.dtype)''')
md("""The training function below is complete. Read it: each step = pick a batch, forward, loss, `zero_grad`, `backward`, `step`. Note `model.train()` at the start (matters for dropout / batchnorm).""")
code('''def train(model, opt, X, y, epochs=5, bs=128):
    loss_fn = nn.CrossEntropyLoss()     # takes LOGITS and integer labels
    model.train()
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=X.device)   # new shuffle each epoch
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())   # .item() -> python float (no graph kept)
    return losses''')
md("**Exercise 5.1** Write `accuracy(model, X, y)`: switch to `eval()` mode, disable gradient tracking with `torch.no_grad()`, return the fraction of correct argmax predictions as a python float.")
ex('''def accuracy(model, X, y):
    # TODO
    return None''',
   '''def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()''')
code('''torch.manual_seed(0)
acc0 = accuracy(MLP().to(device), Xte, yte)
assert 0.0 <= acc0 <= 0.4, acc0       # untrained: around chance (10%)
print("ok, untrained accuracy:", acc0)''')
md("""### Sanity checks before any real training

1. **Loss at init**: with 10 balanced classes and small random logits, the cross-entropy should be about `ln(10) = 2.30`. If it is far from that, something is wrong (bad init, missing normalization, wrong loss).
2. **Overfit one small batch**: a correct model + optimizer must drive the loss on 32 samples to ~0. If it cannot, there is a bug.""")
ex('''torch.manual_seed(0)
model = MLP().to(device)
with torch.no_grad():
    init_loss = None  # TODO: cross-entropy of model on Xtr[:2000], ytr[:2000] (use F.cross_entropy)
print("init loss:", init_loss, "| ln(10) =", np.log(10))''',
   '''torch.manual_seed(0)
model = MLP().to(device)
with torch.no_grad():
    init_loss = F.cross_entropy(model(Xtr[:2000]), ytr[:2000]).item()
print("init loss:", init_loss, "| ln(10) =", np.log(10))''')
code('''assert abs(init_loss - np.log(10)) < 0.3
print("ok")''')
md("**Exercise 5.2** Overfit one batch: take `xb, yb = Xtr[:32], ytr[:32]`, a fresh `MLP`, `Adam(lr=1e-2)`, and run 200 steps on this same batch. Store the final loss in `final_loss`.")
ex('''torch.manual_seed(0)
xb, yb = Xtr[:32], ytr[:32]
m1 = MLP().to(device)
opt = torch.optim.Adam(m1.parameters(), lr=1e-2)
for step in range(200):
    pass  # TODO: forward, loss, zero_grad, backward, step
final_loss = None  # TODO: float''',
   '''torch.manual_seed(0)
xb, yb = Xtr[:32], ytr[:32]
m1 = MLP().to(device)
opt = torch.optim.Adam(m1.parameters(), lr=1e-2)
for step in range(200):
    loss = F.cross_entropy(m1(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
final_loss = loss.item()''')
code('''assert final_loss < 0.01, final_loss
print("ok, final loss", final_loss)''')
md("Now the real training: 5 epochs of Adam on the full training set (should take well under a minute on CPU), then the test accuracy. **Keep `test_acc`: you submit it in section 8.**")
code('''torch.manual_seed(0)
model = MLP(hidden=256).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
losses = train(model, opt, Xtr, ytr, epochs=5)
test_acc = accuracy(model, Xte, yte)
print(f"final train loss (last 100 steps): {np.mean(losses[-100:]):.3f} | test accuracy: {test_acc:.4f} | {time.time()-t0:.0f}s")
assert test_acc > 0.80''')

md("""## 6. `torch.optim`: SGD vs Adam, learning rate

Same model, same data (10,000 samples, 2 epochs), different optimizer and learning rate. The learning rate matters more than almost anything else. Note the optimal range differs: SGD typically wants `lr` around 0.01-0.1 (here), Adam around 1e-4 to 1e-2.""")
code('''Xs, ys = Xtr[:10000], ytr[:10000]
rows = {}
for name, make in {
    "sgd lr=0.001": lambda p: torch.optim.SGD(p, lr=1e-3),
    "sgd lr=0.01":  lambda p: torch.optim.SGD(p, lr=1e-2),
    "sgd lr=0.1":   lambda p: torch.optim.SGD(p, lr=1e-1),
    "adam lr=1e-4": lambda p: torch.optim.Adam(p, lr=1e-4),
    "adam lr=1e-3": lambda p: torch.optim.Adam(p, lr=1e-3),
    "adam lr=1e-2": lambda p: torch.optim.Adam(p, lr=1e-2),
    "adam lr=1e-1": lambda p: torch.optim.Adam(p, lr=1e-1),
}.items():
    torch.manual_seed(0)
    mdl = MLP().to(device)
    ls = train(mdl, make(mdl.parameters()), Xs, ys, epochs=2)
    rows[name] = {"final_loss": np.mean(ls[-20:]), "test_acc": accuracy(mdl, Xte, yte)}
import pandas as pd
df = pd.DataFrame(rows).T.round(3)
df''')
md("**Exercise 6.1** Using `df`, store in `best` the name (index label) of the setting with the highest test accuracy. Then think: why does `adam lr=1e-1` do so badly while `sgd lr=0.1` is fine?")
ex('''best = None  # TODO''',
   '''best = df["test_acc"].idxmax()''')
code('''assert best == df["test_acc"].idxmax()
print("best:", best)''')

md("""## 7. Pitfalls checklist

1. **Softmax before `CrossEntropyLoss`**: the loss already applies log-softmax. Feed it raw logits. (Softmax twice still "trains" but squashes gradients and slows learning.)
2. **Missing `optimizer.zero_grad()`**: gradients accumulate across steps (section 3).
3. **Forgetting `model.eval()`** for evaluation: dropout stays random (and batchnorm uses batch statistics). Switch back with `model.train()` before training again.
4. **Device mismatch**: model and data must be on the same device (`RuntimeError: Expected all tensors to be on the same device`). Move both with `.to(device)`.
5. **Logging with `.item()`**: `losses.append(loss)` keeps the whole graph alive (memory grows); use `loss.item()`.
6. Also: wrong dtype for labels (`CrossEntropyLoss` wants `int64`), not shuffling, not normalizing inputs, evaluating on the training set only.""")
code('''logits = torch.tensor([[2.0, 0.5, -1.0]]); target = torch.tensor([0])
print("correct (logits):         ", F.cross_entropy(logits, target).item())
print("wrong (softmax first):    ", F.cross_entropy(logits.softmax(1), target).item())

d = nn.Sequential(nn.Dropout(0.5)); v = torch.ones(1, 8)
d.train(); print("train mode:", d(v))
d.eval();  print("eval mode: ", d(v))''')
md("**Exercise 7.1** In which mode is the dropout layer `d` right now? Set the variable `mode` to `\"train\"` or `\"eval\"` according to `d.training`, then check by calling it twice: in the right mode the outputs for the same input are identical.")
ex('''d.train()
a1, a2 = d(v), d(v)
mode = None  # TODO: after the call d.eval() below, what should this be?
d.eval()
b1, b2 = d(v), d(v)''',
   '''d.train()
a1, a2 = d(v), d(v)
d.eval()
b1, b2 = d(v), d(v)
mode = "eval" if not d.training else "train"''')
code('''assert mode == "eval" and torch.equal(b1, b2)
print("ok (in train mode a1 and a2 differ:", not torch.equal(a1, a2), ")")''')

md("""## 8. REQUIRED: connect to the class server

This takes ~3 minutes and checks that Colab and the class server work for you. In the labs we will submit results here and compare them across the class.

**Privacy.** Your **campus email** is used **only** to contact competition winners. It is **never shown** to anyone and it is **deleted after the course**. Your **nickname appears on the projector** in class, so **do not use your real name**.

Fill in the two variables (EMAIL must end with `@campus.unimib.it`; nickname: 2-20 characters, letters, digits, `_` or `-`) and run the cells. If the nickname is taken, pick another one. Re-running with the same email and nickname is fine.""")
code('''!wget -q -O aml_client.py {CLIENT_URL}''')
code('''EMAIL = "name.surname@campus.unimib.it"   # <- your campus email
NICKNAME = "my_nickname"                     # <- NOT your real name

assert EMAIL.endswith("@campus.unimib.it"), "use your @campus.unimib.it email"
assert NICKNAME != "my_nickname", "choose a nickname"
assert "test_acc" in globals(), "run section 5 first (it defines test_acc)"

import aml_client as aml
aml.register(EMAIL, NICKNAME)''')
code('''aml.submit("prelab", test_acc=test_acc, device=device, torch=torch.__version__)
print("submitted")''')
code('''import pandas as pd
res = pd.DataFrame(aml.results("prelab"))
print(len(res), "submissions so far")
display(res[["nickname", "test_acc", "device", "torch"]].tail(10))
res["test_acc"].plot.hist(bins=15, title="prelab: test accuracy of the class")''')
md("""If you see your nickname in the table above, you are done. If any cell in this section failed, read the error message (nickname taken, wrong email domain, server down) and fix it; if you cannot, write to the instructor **before** the lab.

See you in the lab!""")


def build(solutions, path):
    nb = nbf.v4.new_notebook()
    nb.metadata["colab"] = {"provenance": []}
    nb.cells = [(nbf.v4.new_markdown_cell if k == "md" else nbf.v4.new_code_cell)(sol if solutions else stu)
                for k, stu, sol in C]
    nbf.write(nb, path)

if __name__ == "__main__":
    import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build(False, "prelab_pytorch.ipynb")
    build(True, "prelab_pytorch_solutions.ipynb")
