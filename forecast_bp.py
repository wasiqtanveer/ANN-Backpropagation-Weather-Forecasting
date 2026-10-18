"""ANN with backpropagation for weather forecasting (Jena climate dataset).

Task: predict the air temperature 6 hours ahead from 4 current measurements:
pressure, temperature, humidity and wind speed.
The network is written from scratch in NumPy (forward pass, backprop, momentum SGD).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(os.path.join(HERE, "weather_4params.csv"))   # hourly readings

HORIZON = 6                                        # hours ahead to forecast
feat = df[["pressure_mbar", "temp_c", "humidity_pct", "wind_speed_ms"]].copy()
target = df["temp_c"].shift(-HORIZON)
ok = feat.notna().all(axis=1) & target.notna()
X, y = feat[ok].values, target[ok].values.reshape(-1, 1)
names = list(feat.columns)

# chronological split: 70% train / 15% validation / 15% test (no shuffling)
n = len(X)
i1, i2 = int(0.70 * n), int(0.85 * n)
Xtr, Xva, Xte = X[:i1], X[i1:i2], X[i2:]
ytr, yva, yte = y[:i1], y[i1:i2], y[i2:]

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8            # scale using train stats only
ymu, ysd = ytr.mean(), ytr.std()
Xtr, Xva, Xte = [(a - mu) / sd for a in (Xtr, Xva, Xte)]
ytr_s, yva_s = (ytr - ymu) / ysd, (yva - ymu) / ysd

# ---------------------------------------------------------------- network
sizes = [X.shape[1], 16, 8, 1]                    # inputs, hidden1, hidden2, output
W = [np.random.randn(a, b) * np.sqrt(2.0 / (a + b)) for a, b in zip(sizes[:-1], sizes[1:])]
b = [np.zeros((1, k)) for k in sizes[1:]]
vW = [np.zeros_like(w) for w in W]
vb = [np.zeros_like(x) for x in b]


def forward(x):
    acts = [x]
    for i in range(len(W)):
        z = acts[-1] @ W[i] + b[i]
        acts.append(np.tanh(z) if i < len(W) - 1 else z)   # linear output (regression)
    return acts


def backward(acts, target_s):
    m = target_s.shape[0]
    delta = (acts[-1] - target_s) / m               # dMSE/dz for linear output (up to factor 2)
    gW, gb = [None] * len(W), [None] * len(W)
    for i in reversed(range(len(W))):
        gW[i] = acts[i].T @ delta
        gb[i] = delta.sum(0, keepdims=True)
        if i > 0:
            delta = (delta @ W[i].T) * (1 - acts[i] ** 2)   # tanh'(z) = 1 - a^2
    return gW, gb


LR, MOM, BATCH, EPOCHS = 0.01, 0.9, 128, 60
train_loss = [np.mean((forward(Xtr)[-1] - ytr_s) ** 2)]   # error before any training
val_loss = [np.mean((forward(Xva)[-1] - yva_s) ** 2)]
print(f"epoch   0  train MSE {train_loss[-1]:.4f}  val MSE {val_loss[-1]:.4f}")
for ep in range(EPOCHS):
    idx = np.random.permutation(len(Xtr))
    for s in range(0, len(idx), BATCH):
        j = idx[s:s + BATCH]
        gW, gb = backward(forward(Xtr[j]), ytr_s[j])
        for i in range(len(W)):
            vW[i] = MOM * vW[i] - LR * gW[i]
            vb[i] = MOM * vb[i] - LR * gb[i]
            W[i] += vW[i]
            b[i] += vb[i]
    train_loss.append(np.mean((forward(Xtr)[-1] - ytr_s) ** 2))
    val_loss.append(np.mean((forward(Xva)[-1] - yva_s) ** 2))
    if ep % 10 == 0 or ep == EPOCHS - 1:
        print(f"epoch {ep + 1:3d}  train MSE {train_loss[-1]:.4f}  val MSE {val_loss[-1]:.4f}")

# ---------------------------------------------------------------- evaluation
pred = forward(Xte)[-1] * ysd + ymu
persist = Xte[:, names.index("temp_c")].reshape(-1, 1) * sd[names.index("temp_c")] + mu[names.index("temp_c")]


def metrics(true, p):
    err = p - true
    return np.mean(err ** 2), np.sqrt(np.mean(err ** 2)), np.mean(np.abs(err)), \
        1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)


mse, rmse, mae, r2 = metrics(yte, pred)
pmse, prmse, pmae, pr2 = metrics(yte, persist)
res = (
    f"Samples: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}\n"
    f"Inputs ({len(names)}): {', '.join(names)}\n"
    f"Architecture: {sizes}, tanh hidden, linear output, lr={LR}, momentum={MOM}, "
    f"batch={BATCH}, epochs={EPOCHS}\n"
    f"Forecast horizon: {HORIZON} hours\n\n"
    f"{'Model':<28}{'MSE':>8}{'RMSE':>8}{'MAE':>8}{'R2':>8}\n"
    f"{'ANN (backprop)':<28}{mse:8.3f}{rmse:8.3f}{mae:8.3f}{r2:8.3f}\n"
    f"{'Baseline (same as now)':<28}{pmse:8.3f}{prmse:8.3f}{pmae:8.3f}{pr2:8.3f}\n"
)
print("\n" + res)
open(os.path.join(OUT, "results.txt"), "w").write(res)

# ---------------------------------------------------------------- plots
plt.figure(figsize=(7, 4))
plt.plot(range(EPOCHS + 1), train_loss, label="Training")
plt.plot(range(EPOCHS + 1), val_loss, label="Validation")
plt.xlabel("Iteration (epoch)"); plt.ylabel("MSE (scaled)")
plt.title("Error vs iteration"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "1_error_vs_iteration.png"), dpi=150); plt.close()

k = 300
plt.figure(figsize=(9, 4))
plt.plot(yte[:k], label="Actual", lw=1.5)
plt.plot(pred[:k], label="ANN forecast", lw=1.5)
plt.xlabel("Test hour"); plt.ylabel("Temperature (°C)")
plt.title(f"Actual vs forecast ({HORIZON} h ahead, first {k} test hours)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "2_actual_vs_forecast.png"), dpi=150); plt.close()

plt.figure(figsize=(5, 5))
plt.scatter(yte, pred, s=3, alpha=.3)
lo, hi = yte.min(), yte.max()
plt.plot([lo, hi], [lo, hi], "r--", label="Perfect")
plt.xlabel("Actual (°C)"); plt.ylabel("Predicted (°C)")
plt.title(f"Predicted vs actual, R² = {r2:.3f}"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "3_scatter.png"), dpi=150); plt.close()

plt.figure(figsize=(6, 4))
plt.hist((pred - yte).ravel(), bins=60)
plt.xlabel("Forecast error (°C)"); plt.ylabel("Count")
plt.title("Error distribution"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "4_error_hist.png"), dpi=150); plt.close()
print("Saved plots and results.txt in the outputs folder")
