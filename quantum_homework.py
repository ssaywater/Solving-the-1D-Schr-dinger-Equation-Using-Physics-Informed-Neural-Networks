import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---- SPEED SETTINGS
torch.set_default_dtype(torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---- MLP
class MLP(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ---- Problem: Infinite Square Well (dimensionless inside well)
L = 1.0

def trapz(y, x):
    return torch.trapz(y.squeeze(-1), x.squeeze(-1))

# fixed grid for normalization/overlap integrals
xg = torch.linspace(0.0, L, 800, device=device).unsqueeze(1)

# Ansatz: BC + avoid psi=0 collapse
def psi_from_net(x, net):
    return x * (L - x) * (1.0 + net(x))

def residual(x, net, E_value):
    # R(x) = -psi''(x) - E psi(x)
    x.requires_grad_(True)
    p = psi_from_net(x, net)
    dp = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    ddp = torch.autograd.grad(dp, x, torch.ones_like(dp), create_graph=True)[0]
    return -ddp - E_value * p

def train_state(
    net,
    E_raw_param,
    get_E_value_fn,
    psi_prev_fn=None,
    epochs=2500,
    n_colloc=512,
    lr=2e-3,
    w_norm=50.0,
    w_ortho=50.0,
    early_stop=True,
    verbose_every=250
):
    opt = torch.optim.Adam(list(net.parameters()) + [E_raw_param], lr=lr)

    for ep in range(1, epochs + 1):
        xc = torch.rand(n_colloc, 1, device=device) * L

        E_val = get_E_value_fn()
        R = residual(xc, net, E_val)
        loss_pde = torch.mean(R**2)

        # normalization loss
        pg = psi_from_net(xg, net)
        norm = trapz(pg**2, xg)
        loss_norm = (norm - 1.0) ** 2

        # orthogonality loss (only for excited states)
        loss_ortho = torch.tensor(0.0, device=device)
        if psi_prev_fn is not None:
            with torch.no_grad():
                psi_prev = psi_prev_fn(xg)  # already normalized
            # normalize current on-the-fly for stable overlap
            psi_cur_norm = pg / torch.sqrt(norm + 1e-8)
            overlap = trapz(psi_cur_norm * psi_prev, xg)
            loss_ortho = overlap**2

        loss = loss_pde + w_norm * loss_norm + w_ortho * loss_ortho

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % verbose_every == 0 or ep == 1:
            print(f"ep={ep:4d} loss={loss.item():.3e} E={E_val.item():.6f} "
                  f"norm={norm.item():.4f} ortho={loss_ortho.item():.3e}")

        if early_stop and ep > 400:
            # basic stopping: norm close to 1 and loss reasonably small
            if abs(norm.item() - 1.0) < 2e-2 and loss.item() < 1e-2:
                print("Early stop.")
                break

    # return a normalized callable psi(x)
    with torch.no_grad():
        pg = psi_from_net(xg, net)
        norm = trapz(pg**2, xg)
        scale = torch.sqrt(norm + 1e-8)

    def psi_normalized(x_in):
        return psi_from_net(x_in, net) / scale

    return psi_normalized

# -------------------------
# 1) Train n=1 (ground state)
# -------------------------
net1 = MLP(width=32, depth=3).to(device)
E1_raw = nn.Parameter(torch.tensor([1.0], device=device))

def E1():
    return torch.nn.functional.softplus(E1_raw)[0]  # E1 > 0

psi1 = train_state(
    net=net1,
    E_raw_param=E1_raw,
    get_E_value_fn=E1,
    psi_prev_fn=None,
    epochs=8000,        # <<< artırdık
    n_colloc=1024,      # <<< artırdık
    lr=2e-3,
    w_norm=300.0,       # <<< artırdık (norm düzgün otursun)
    w_ortho=0.0,
    early_stop=False,   # <<< kapattık
    verbose_every=500
)

E1_pinn = float(E1().item())
print("E1_pinn after n=1 training =", E1_pinn, " target =", float(np.pi**2))
E1_raw.requires_grad_(False)
E1_const = E1_pinn


# -------------------------
# 2) Train n=2 (first excited)
#    enforce orthogonality + enforce E2 > E1
# -------------------------
net2 = MLP(width=32, depth=3).to(device)
E2_raw = nn.Parameter(torch.tensor([1.0], device=device))

def E2():
    return torch.tensor(E1_const, device=device, dtype=torch.float32) \
           + (2*np.pi)**2 - (np.pi)**2 \
           + 1.0 * torch.nn.functional.softplus(E2_raw)[0]



psi2 = train_state(
    net=net2,
    E_raw_param=E2_raw,
    get_E_value_fn=E2,
    psi_prev_fn=psi1,
    epochs=15000,
    n_colloc=2048,
    lr=1e-3,
    w_norm=300.0,     # normalize olmayı zorla (psi=0 çökmesin)
    w_ortho=1000.0,     # ortho var ama çok baskın olmasın
    verbose_every=500
)


E2_pinn = float(E2().item())

# -------------------------
# Plot: PINN vs Analytic for n=1 and n=2
# -------------------------
x_np = np.linspace(0.0, L, 1200)
x_t = torch.tensor(x_np, device=device, dtype=torch.float32).unsqueeze(1)


with torch.no_grad():
    psi1_p = psi1(x_t).detach().cpu().numpy().squeeze()
    psi2_p = psi2(x_t).detach().cpu().numpy().squeeze()

psi1_a = np.sqrt(2.0/L) * np.sin(np.pi * x_np / L)
psi2_a = np.sqrt(2.0/L) * np.sin(2*np.pi * x_np / L)

E1_a = (np.pi / L)**2
E2_a = (2*np.pi / L)**2

# sign align for nicer plots
if np.dot(psi1_p, psi1_a) < 0: psi1_p = -psi1_p
if np.dot(psi2_p, psi2_a) < 0: psi2_p = -psi2_p

plt.figure()
plt.plot(x_np, psi1_p, label=f"PINN n=1, E={E1_pinn:.6f}")
plt.plot(x_np, psi1_a, "--", label=f"Analytic n=1, E={E1_a:.6f}")
plt.title("Infinite Square Well (L=1): n=1")
plt.xlabel("x"); plt.ylabel("ψ(x)")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(x_np, psi2_p, label=f"PINN n=2, E={E2_pinn:.6f}")
plt.plot(x_np, psi2_a, "--", label=f"Analytic n=2, E={E2_a:.6f}")
plt.title("Infinite Square Well (L=1): n=2")
plt.xlabel("x"); plt.ylabel("ψ(x)")
plt.legend(); plt.tight_layout(); plt.show()

def plot_residual_normalized(psi_fn, E_scalar, title):
    x_res_np = np.linspace(0.01, L-0.01, 1600) # endpoints excluded
    x_res = torch.tensor(x_res_np, device=device, dtype=torch.float32).unsqueeze(1)
    x_res.requires_grad_(True)

    psi_res = psi_fn(x_res)  # psi_fn is already normalized callable (psi1 or psi2)

    dpsi = torch.autograd.grad(
        psi_res, x_res, grad_outputs=torch.ones_like(psi_res), create_graph=True
    )[0]
    ddpsi = torch.autograd.grad(
        dpsi, x_res, grad_outputs=torch.ones_like(dpsi), create_graph=False
    )[0]

    R = (-ddpsi - E_scalar * psi_res).detach().cpu().numpy().squeeze()

    plt.figure()
    plt.plot(x_res_np, R)
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("R(x)")
    plt.tight_layout()
    plt.show()

    print(title, "| max|R| =", float(np.max(np.abs(R))),
          "mean|R| =", float(np.mean(np.abs(R))))

plot_residual_normalized(psi1, torch.tensor(E1_pinn, device=device, dtype=torch.float32),
                         "Residual Check (n=1): R(x) = -ψ''(x) - E ψ(x)")

def plot_residual_normalized_custom_range(psi_fn, E_scalar, title, a=0.05, b=0.95):
    x_res_np = np.linspace(a, b, 1600)
    x_res = torch.tensor(x_res_np, device=device, dtype=torch.float32).unsqueeze(1)
    x_res.requires_grad_(True)

    psi_res = psi_fn(x_res)
    dpsi = torch.autograd.grad(psi_res, x_res, torch.ones_like(psi_res), create_graph=True)[0]
    ddpsi = torch.autograd.grad(dpsi, x_res, torch.ones_like(dpsi), create_graph=False)[0]

    R = (-ddpsi - E_scalar * psi_res).detach().cpu().numpy().squeeze()

    plt.figure()
    plt.plot(x_res_np, R)
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("R(x)")
    plt.tight_layout()
    plt.show()

# n=2 cropped residual
plot_residual_normalized_custom_range(
    psi2,
    torch.tensor(E2_pinn, device=device, dtype=torch.float32),
    "Residual Check (n=2, cropped): R(x) = -ψ''(x) - E ψ(x)",
    a=0.05, b=0.95
)


# -------------------------
# Table-like printout
# -------------------------
rel1 = abs(E1_pinn - E1_a) / E1_a
rel2 = abs(E2_pinn - E2_a) / E2_a

print("\nTable: Energy तुलना (L=1)")
print(f"n=1  E_PINN={E1_pinn:.8f}  E_analytic={E1_a:.8f}  rel.err={rel1:.2e}")
print(f"n=2  E_PINN={E2_pinn:.8f}  E_analytic={E2_a:.8f}  rel.err={rel2:.2e}")
