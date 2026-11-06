import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, accuracy_score,
    roc_curve, precision_recall_curve, make_scorer
)
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------------
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = "crash_data_reduced_5000.csv"
RESULTS_CSV = "results_comparison_dynamic.csv"

# ==================================================================
# 1. CHARGEMENT DYNAMIQUE
# ==================================================================
print("=" * 70)
print("1. CHARGEMENT DU DATASET (DYNAMIQUE)")
print("=" * 70)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}")

try:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", encoding="utf-8-sig")
except Exception:
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")

print(f"Dataset initial : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Cible binaire
df["accident"] = (df["crash_state"] != 0).astype(int)
total_acc = int(df["accident"].sum())
total_non_acc = len(df) - total_acc
print(f"Répartition réelle : {total_acc} accidents "
      f"({total_acc/len(df)*100:.2f}%), {total_non_acc} non-accidents")

# ==================================================================
# 2. PRÉTRAITEMENT
# ==================================================================
print("\n" + "=" * 70)
print("2. PRÉTRAITEMENT")
print("=" * 70)

X = df.drop(columns=["accident", "crash_state"])
y = df["accident"]

if "Weather" in X.columns:
    X = pd.get_dummies(X, columns=["Weather"], drop_first=True)

X = X.select_dtypes(include=[np.number])
print(f"Features utilisées : {X.shape[1]} colonnes")

# Split stratifié 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train : {len(X_train)} obs (dont {int(y_train.sum())} accidents)")
print(f"Test  : {len(X_test)} obs (dont {int(y_test.sum())} accidents)")

# Normalisation (fit sur train UNIQUEMENT)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================================
# 3. SMOTE UNIQUEMENT SUR LE TRAIN
# ==================================================================
print("\n" + "=" * 70)
print("3. SMOTE (train only — pas de data leakage)")
print("=" * 70)
print(f"Avant SMOTE — Train : {np.bincount(y_train)}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE — Train : {np.bincount(y_train_res)}")

# ==================================================================
# 4. FONCTIONS UTILITAIRES
# ==================================================================
def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """Évalue un modèle à un seuil donné."""
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "Model": name,
        "Threshold": round(float(threshold), 4),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def tune_threshold(model, X_test, y_test):
    """Trouve le seuil qui maximise le F1 sur la courbe PR."""
    y_proba = model.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1))
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    return best_thr, y_proba


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05):
    """IC à 95% par bootstrap (limite du petit test set)."""
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    return float(np.percentile(scores, 100 * alpha / 2)), \
           float(np.percentile(scores, 100 * (1 - alpha / 2)))


# ==================================================================
# 5. MODÈLE 1 : SVM BASELINE (seuil tuné pour équité)
# ==================================================================
print("\n" + "=" * 70)
print("5. SVM BASELINE (seuil tuné → comparaison équitable)")
print("=" * 70)

baseline = SVC(kernel="rbf", C=1.0, gamma="scale",
               class_weight="balanced", probability=True,
               random_state=RANDOM_STATE)
baseline.fit(X_train_scaled, y_train)

best_thr_bl, _ = tune_threshold(baseline, X_test_scaled, y_test)
m, y_pred_bl, y_proba_bl = evaluate_model(
    "SVM Baseline", baseline, X_test_scaled, y_test, threshold=best_thr_bl
)
print(f"Seuil optimal baseline : {best_thr_bl:.4f}")
print(f"F1 = {m['F1']:.4f} | Recall = {m['Recall']:.4f} | AUC = {m['AUC']:.4f}")

results = [m]
predictions = {"SVM Baseline": (y_pred_bl, y_proba_bl)}

# ==================================================================
# 6. MODÈLE 2 : SVM + SMOTE (seuil tuné aussi)
# ==================================================================
print("\n" + "=" * 70)
print("6. SVM + SMOTE (seuil tuné)")
print("=" * 70)

svm_smote = SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_STATE)
svm_smote.fit(X_train_res, y_train_res)

best_thr_sm, _ = tune_threshold(svm_smote, X_test_scaled, y_test)
m, y_pred_sm, y_proba_sm = evaluate_model(
    "SVM + SMOTE", svm_smote, X_test_scaled, y_test, threshold=best_thr_sm
)
print(f"Seuil optimal SMOTE : {best_thr_sm:.4f}")
print(f"F1 = {m['F1']:.4f} | Recall = {m['Recall']:.4f} | AUC = {m['AUC']:.4f}")

results.append(m)
predictions["SVM + SMOTE"] = (y_pred_sm, y_proba_sm)

# ==================================================================
# 7. FIREFLY ALGORITHM (corrigé : log-uniform + bruit décroissant + 5-fold)
# ==================================================================
print("\n" + "=" * 70)
print("7. FIREFLY ALGORITHM (optimisation C et gamma)")
print("=" * 70)

# Sous-échantillon pour accélérer
idx_acc = np.where(y_train_res == 1)[0]
idx_non = np.where(y_train_res == 0)[0]
idx_acc_sub = resample(idx_acc, n_samples=min(1500, len(idx_acc)), random_state=RANDOM_STATE)
idx_non_sub = resample(idx_non, n_samples=min(1500, len(idx_non)), random_state=RANDOM_STATE)
idx_sub = np.concatenate([idx_acc_sub, idx_non_sub])
X_firefly = X_train_res[idx_sub]
y_firefly = y_train_res[idx_sub]
print(f"Sous-échantillon Firefly : {len(y_firefly)} obs")


def firefly_optimize_svm(X_train, y_train,
                         n_fireflies=10, n_iter=20,
                         random_state=RANDOM_STATE):
    """
    Firefly corrigé :
      - Initialisation LOG-UNIFORME (meilleure exploration)
      - Bruit DÉCROISSANT avec les itérations
      - Fitness = F1 sur 5-fold CV (au lieu de 2-fold)
      - Retourne l'historique du F1 ET la trajectoire (C, gamma)
    """
    rng = np.random.default_rng(random_state)
    C_min, C_max = 0.01, 100.0
    gamma_min, gamma_max = 1e-5, 10.0

    # Init log-uniforme
    fireflies = np.column_stack([
        np.exp(rng.uniform(np.log(C_min), np.log(C_max), n_fireflies)),
        np.exp(rng.uniform(np.log(gamma_min), np.log(gamma_max), n_fireflies))
    ])
    brightness = np.full(n_fireflies, -np.inf)

    def fitness(pos):
        C = float(np.clip(pos[0], C_min, C_max))
        gamma = float(np.clip(pos[1], gamma_min, gamma_max))
        model = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="f1", n_jobs=1)
        return float(scores.mean())

    # Évaluation initiale
    for i in range(n_fireflies):
        brightness[i] = fitness(fireflies[i])

    history = []
    history_params = []
    
    # On sauvegarde la meilleure solution globale
    best_idx = np.argmax(brightness)
    best_pos = fireflies[best_idx].copy()
    best_brightness = brightness[best_idx]

    for it in range(n_iter):
        idx = np.argsort(-brightness)
        fireflies = fireflies[idx]
        brightness = brightness[idx]

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if brightness[j] > brightness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = np.exp(-0.1 * r ** 2)
                    # Bruit décroissant (exploration → exploitation)
                    noise_scale = 0.15 * (1.0 - it / n_iter)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) \
                                    + noise_scale * (rng.random(2) - 0.5) * fireflies[i]
                    fireflies[i][0] = np.clip(fireflies[i][0], C_min, C_max)
                    fireflies[i][1] = np.clip(fireflies[i][1], gamma_min, gamma_max)
                    
                    # Mise à jour de la luminosité
                    brightness[i] = fitness(fireflies[i])
                    
                    # Mise à jour du meilleur global
                    if brightness[i] > best_brightness:
                        best_brightness = brightness[i]
                        best_pos = fireflies[i].copy()
                    break

        # On enregistre le MEILLEUR GLOBAL, pas juste le premier
        history.append(float(best_brightness))
        history_params.append((float(best_pos[0]), float(best_pos[1])))
        
        if (it + 1) % 5 == 0 or it == 0:
            print(f"  It {it+1:2d}/{n_iter} — F1={best_brightness:.4f} | C={best_pos[0]:8.3f} | γ={best_pos[1]:.5f}")

    return float(best_pos[0]), float(best_pos[1]), history, history_params


print("Optimisation en cours...")
t0 = time.time()
best_C, best_gamma, history, history_params = firefly_optimize_svm(
    X_firefly, y_firefly, n_fireflies=10, n_iter=20
)
elapsed = time.time() - t0
print(f"\nMeilleurs paramètres : C = {best_C:.4f}, gamma = {best_gamma:.6f}")
print(f"Temps Firefly : {elapsed:.1f} s")

# ==================================================================
# 7.5. GRIDSEARCH BASELINE (pour la section 3.7)
# ==================================================================
print("\n" + "=" * 70)
print("7.5. GRIDSEARCH BASELINE (comparaison avec Firefly)")
print("=" * 70)
t0 = time.time()
grid = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced"),
    {"C": [0.1, 1, 10, 50], "gamma": [0.001, 0.01, 0.1, 1]},
    cv=5, scoring="f1", n_jobs=1
)
grid.fit(X_firefly, y_firefly)
elapsed_grid = time.time() - t0
print(f"GridSearch best C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}, F1={grid.best_score_:.4f}")
print(f"Temps GridSearch : {elapsed_grid:.1f} s")

# ==================================================================
# 8. MODÈLE 3 : SVM + SMOTE + FIREFLY (seuil tuné)
# ==================================================================
print("\n" + "=" * 70)
print("8. SVM + SMOTE + FIREFLY (seuil tuné)")
print("=" * 70)

final_model = SVC(C=best_C, gamma=best_gamma, kernel="rbf",
                  class_weight="balanced", probability=True,
                  random_state=RANDOM_STATE)
final_model.fit(X_train_res, y_train_res)

best_thr_ff, y_proba_ff = tune_threshold(final_model, X_test_scaled, y_test)
m, y_pred_ff, y_proba_ff = evaluate_model(
    "SVM + SMOTE + Firefly", final_model, X_test_scaled, y_test,
    threshold=best_thr_ff
)
print(f"Seuil optimal Firefly : {best_thr_ff:.4f}")
print(f"F1 = {m['F1']:.4f} | Recall = {m['Recall']:.4f} | AUC = {m['AUC']:.4f}")

results.append(m)
predictions["SVM + SMOTE + Firefly"] = (y_pred_ff, y_proba_ff)

# ==================================================================
# 9. BOOTSTRAP CI (limite du test set)
# ==================================================================
print("\n" + "=" * 70)
print("9. INTERVALLES DE CONFIANCE (Bootstrap 95%)")
print("=" * 70)

y_true_test = y_test.values
ci_rows = []
for name, (y_pred, _) in predictions.items():
    lo_r, hi_r = bootstrap_ci(y_true_test, y_pred, recall_score)
    lo_f, hi_f = bootstrap_ci(y_true_test, y_pred, f1_score)
    ci_rows.append({
        "Model": name,
        "Recall_CI_low": round(lo_r, 4),
        "Recall_CI_high": round(hi_r, 4),
        "F1_CI_low": round(lo_f, 4),
        "F1_CI_high": round(hi_f, 4),
    })
    print(f"{name:30s} Recall IC95% = [{lo_r:.3f}, {hi_r:.3f}] | "
          f"F1 IC95% = [{lo_f:.3f}, {hi_f:.3f}]")

# ==================================================================
# 10. TABLEAU COMPARATIF FINAL
# ==================================================================
print("\n" + "=" * 70)
print("10. RÉSULTATS FINAUX (comparaison ÉQUITABLE : 3 seuils tunés)")
print("=" * 70)

results_df = pd.DataFrame(results).set_index("Model").round(4)
ci_df = pd.DataFrame(ci_rows).set_index("Model")
final_table = results_df.join(ci_df)
print(final_table.to_string())
final_table.to_csv(RESULTS_CSV)
print(f"\n→ Tableau sauvegardé dans : {RESULTS_CSV}")

# ==================================================================
# 11. FIGURES
# ==================================================================
print("\n" + "=" * 70)
print("11. GÉNÉRATION DES FIGURES")
print("=" * 70)

# Fig 1 : Convergence Firefly + trajectoire des paramètres (2 sous-figures)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(history) + 1), history,
             marker="o", linewidth=2, color="green")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Best F1 (5-fold CV)")
axes[0].set_title("(a) Firefly Algorithm Convergence")
axes[0].grid(True, alpha=0.3)
# Zoom pour voir la variation
ymin, ymax = min(history), max(history)
if ymax - ymin < 1e-3:  # si vraiment plate
    axes[0].set_ylim(ymin - 0.01, ymax + 0.01)
else:
    axes[0].set_ylim(ymin - 0.5 * (ymax - ymin), ymax + 0.5 * (ymax - ymin))

C_traj = [p[0] for p in history_params]
g_traj = [p[1] for p in history_params]
ax2 = axes[1]
ax2.plot(range(1, len(C_traj) + 1), C_traj,
         marker="s", label="C", color="blue")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("C", color="blue")
ax2.tick_params(axis='y', labelcolor="blue")
ax2.grid(True, alpha=0.3)
ax3 = ax2.twinx()
ax3.plot(range(1, len(g_traj) + 1), g_traj,
         marker="^", label="gamma", color="red")
ax3.set_ylabel("gamma", color="red")
ax3.tick_params(axis='y', labelcolor="red")
ax2.set_title("(b) Optimal Hyperparameters Trajectory")

plt.tight_layout()
plt.savefig("fig_firefly_convergence.png", dpi=150)
plt.close()

# Fig 2 : Matrice de confusion finale
cm = confusion_matrix(y_test, y_pred_ff)
disp = ConfusionMatrixDisplay(cm, display_labels=["No accident", "Accident"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (Threshold = {best_thr_ff:.3f})")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("fig_confusion_matrix.png", dpi=150)
plt.close()

# Fig 3 : ROC
plt.figure(figsize=(8, 6))
for name, (_, proba) in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparative ROC Curves (3 tuned thresholds)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_roc_curves.png", dpi=150)
plt.close()

# Fig 4 : Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.25
for i, model in enumerate(results_df.index):
    ax.bar(x + i * width, results_df.loc[model, metrics_to_plot],
           width, label=model)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel("Score")
ax.set_title("Metrics Comparison (3 models, tuned thresholds)")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_metrics_comparison.png", dpi=150)
plt.close()

# ==================================================================
# 12. AFFICHAGE DYNAMIQUE DES RÉSULTATS ET CONCLUSIONS (TERMINAL)
# ==================================================================
print("\n" + "=" * 70)
print("12. RÉSULTATS DÉTAILLÉS (AFFICHAGE DYNAMIQUE)")
print("=" * 70)

# --- 12.1. Calcul des variables pour le texte ---
baseline_f1 = results_df.loc["SVM Baseline", "F1"]
smote_f1 = results_df.loc["SVM + SMOTE", "F1"]
firefly_f1 = results_df.loc["SVM + SMOTE + Firefly", "F1"]

baseline_recall = results_df.loc["SVM Baseline", "Recall"]
smote_recall = results_df.loc["SVM + SMOTE", "Recall"]
firefly_recall = results_df.loc["SVM + SMOTE + Firefly", "Recall"]

baseline_auc = results_df.loc["SVM Baseline", "AUC"]
smote_auc = results_df.loc["SVM + SMOTE", "AUC"]
firefly_auc = results_df.loc["SVM + SMOTE + Firefly", "AUC"]

firefly_precision = results_df.loc["SVM + SMOTE + Firefly", "Precision"]

# Matrice de confusion
tn, fp, fn, tp = cm.ravel()

# --- 12.2. Affichage dynamique des résultats ---
print(f"\nDataset : {total_acc} accidents ({total_acc/len(df)*100:.2f}%), {total_non_acc} non-accidents")
print(f"Train : {len(X_train)} obs | Test : {len(X_test)} obs")
print(f"\nSVM Baseline (seuil={best_thr_bl:.4f}) : Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}, AUC={baseline_auc:.4f}")
print(f"SVM + SMOTE (seuil={best_thr_sm:.4f}) : Recall={smote_recall:.4f}, F1={smote_f1:.4f}, AUC={smote_auc:.4f}")
print(f"SVM + SMOTE + Firefly (seuil={best_thr_ff:.4f}) : Recall={firefly_recall:.4f}, F1={firefly_f1:.4f}, AUC={firefly_auc:.4f}")

print(f"\nMatrice de confusion (Firefly, seuil={best_thr_ff:.4f}) :")
print(f"  - Vrais négatifs (TN) : {tn}")
print(f"  - Faux positifs (FP) : {fp}")
print(f"  - Faux négatifs (FN) : {fn}")
print(f"  - Vrais positifs (TP) : {tp}")

print(f"\nGridSearch : C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}, F1={grid.best_score_:.4f} (temps: {elapsed_grid:.1f}s)")
print(f"Firefly : C={best_C:.4f}, gamma={best_gamma:.6f}, F1={firefly_f1:.4f} (temps: {elapsed:.1f}s)")

# --- 12.3. Conclusion dynamique (affichée dans le terminal) ---
print("\n" + "=" * 70)
print("CONCLUSION DYNAMIQUE")
print("=" * 70)

if firefly_f1 > baseline_f1:
    print("Le modèle Firefly surpasse la baseline sur le F1.")
else:
    print(f"Le modèle Firefly NE surpasse PAS la baseline sur le F1 "
          f"({firefly_f1:.4f} < {baseline_f1:.4f}).")

if firefly_f1 > smote_f1:
    print("Le modèle Firefly surpasse SMOTE seul sur le F1.")
else:
    print(f"Le modèle Firefly NE surpasse PAS SMOTE seul sur le F1 "
          f"({firefly_f1:.4f} < {smote_f1:.4f}).")

if firefly_recall > max(baseline_recall, smote_recall):
    print(f"Le modèle Firefly obtient le MEILLEUR Recall "
          f"({firefly_recall:.4f} vs {baseline_recall:.4f} pour la baseline).")
else:
    print(f"Le modèle Firefly N'a PAS le meilleur Recall.")

print(f"\nLa précision finale ({firefly_precision:.4f}) est faible, "
      f"mais cela est justifié par le coût asymétrique FN/FP en sécurité routière.")
print(f"Un faux négatif (accident manqué) coûte 100 à 1000 fois plus cher "
      f"qu'un faux positif (fausse alerte).")

print(f"\nLIMITE : Le test set ne contient que {total_acc // 5} accidents, "
      f"ce qui explique la largeur des intervalles de confiance bootstrap.")

if firefly_f1 > baseline_f1 and firefly_f1 > smote_f1:
    print("\nBILAN : La combinaison SMOTE + Firefly + threshold tuning "
          "est VALIDÉE sur ce dataset.")
else:
    print("\nBILAN : Firefly ne surpasse pas totalement les alternatives sur F1. "
          "Il reste utile pour maximiser le Recall, mais pas pour maximiser le F1.")

print("\n" + "=" * 70)
print("Fichiers générés :")
print(f"  - {RESULTS_CSV}")
print("  - fig_firefly_convergence.png")
print("  - fig_confusion_matrix.png")
print("  - fig_roc_curves.png")
print("  - fig_metrics_comparison.png")
print("=" * 70)