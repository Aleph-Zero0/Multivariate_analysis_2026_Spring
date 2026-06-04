# ESOL aqueous solubility - multivariate analysis (code appendix)

# 0. Setup - load data and compute descriptors from SMILES
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

sns.set_theme(style="whitegrid", context="notebook")
RNG = 42  # global seed (LASSO CV, train/test splits)

raw = pd.read_csv("delaney.csv").rename(columns={
    "measured log(solubility:mol/L)": "logS",
    "ESOL predicted log(solubility:mol/L)": "esol_pred",
})
print("loaded:", raw.shape)
raw.head(3)
# Compute 8 physicochemical descriptors from each SMILES.
DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
               "NumRotatableBonds", "NumAromaticRings", "FractionCSP3"]

def compute_descriptors(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(m),
        "MolLogP": Crippen.MolLogP(m),                       # lipophilicity
        "TPSA": rdMolDescriptors.CalcTPSA(m),                # polar surface area
        "NumHDonors": rdMolDescriptors.CalcNumHBD(m),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(m),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(m),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(m),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(m),
    }

desc = raw["SMILES"].apply(compute_descriptors)
print("SMILES that failed to parse:", desc.isna().sum())

data = pd.concat([raw[["logS"]], pd.DataFrame(list(desc))], axis=1).dropna().reset_index(drop=True)

# All analysis uses the full dataset.
df = data
X = df[DESCRIPTORS]
y = df["logS"]
print("working dataset:", df.shape)
df.head(3)

# 1. The dataset
print(f"molecules: {len(data)}   quantitative variables: {len(DESCRIPTORS)} descriptors + logS")
for d in DESCRIPTORS: print(" -", d)

# 2. Description and why it is interesting
summary = X.describe().T[["mean", "std", "min", "max"]]
summary["skew"] = X.skew()
summary.round(2)

# 3. Pairwise scatter-plot matrix
g = sns.pairplot(df[DESCRIPTORS + ["logS"]], corner=True,
                 plot_kws=dict(s=6, alpha=0.2, edgecolor="none"),
                 diag_kws=dict(fill=True))
g.fig.suptitle(f"Pairwise scatter matrix (n={len(df)})", y=1.01)
plt.show()

# 4. Are the data Gaussian?
import scipy.stats as stats
cols = DESCRIPTORS + ["logS"]
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, c in zip(axes.ravel(), cols):
    sns.histplot(df[c], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"{c}  (skew={df[c].skew():.2f})")
fig.suptitle(f"Marginal distributions (n={len(df)})", y=1.005); plt.tight_layout(); plt.show()
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, c in zip(axes.ravel(), cols):
    stats.probplot(df[c], dist="norm", plot=ax); ax.set_title(c)
fig.suptitle(f"Normal QQ-plots (n={len(df)})", y=1.005); plt.tight_layout(); plt.show()
# Shapiro-Wilk normality test per variable
pd.DataFrame({c: stats.shapiro(df[c])[1] for c in cols}, index=["Shapiro p-value"]).T.round(4)

# 5. Choice of response variable

# 6. Linear regression
import statsmodels.api as sm
Xc = sm.add_constant(X)
ols = sm.OLS(y, Xc).fit()
print(ols.summary())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(ols.fittedvalues, y, s=12, alpha=0.5)
lims = [y.min(), y.max()]; ax[0].plot(lims, lims, "r--")
ax[0].set(xlabel="predicted logS", ylabel="observed logS", title=f"Fit  (R²={ols.rsquared:.3f})")
ax[1].scatter(ols.fittedvalues, ols.resid, s=12, alpha=0.5); ax[1].axhline(0, color="r", ls="--")
ax[1].set(xlabel="predicted logS", ylabel="residual", title="Residuals")
plt.tight_layout(); plt.show()
# condition number: raw scale vs standardized predictors
Xz = (X - X.mean()) / X.std()
print(f"condition number, raw predictors          : {ols.condition_number:8.1f}")
print(f"condition number, standardized predictors  : {np.linalg.cond(Xz.values):8.2f}")

# 7. Variable selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({"VIF": [variance_inflation_factor(Xc.values, i)
                            for i in range(Xc.shape[1])]}, index=Xc.columns).drop("const")
vif.round(2)
# Best-subset selection by AIC over all 2^8-1 = 255 subsets (exhaustive).
import itertools
candidates = []
for k in range(1, len(DESCRIPTORS) + 1):
    for combo in itertools.combinations(DESCRIPTORS, k):
        aic = sm.OLS(y, sm.add_constant(X[list(combo)])).fit().aic
        candidates.append((aic, combo))
best_aic, chosen = min(candidates, key=lambda t: t[0])
chosen = list(chosen)
full_aic = sm.OLS(y, sm.add_constant(X)).fit().aic
best_r2 = sm.OLS(y, sm.add_constant(X[chosen])).fit().rsquared
print(f"best-subset AIC = {best_aic:.2f}  (full 8-var model AIC = {full_aic:.2f})")
print(f"selected {len(chosen)} vars: {chosen}")
print(f"R²: best-subset = {best_r2:.3f}  vs  full = {ols.rsquared:.3f}")
sm.OLS(y, sm.add_constant(X[chosen])).fit().summary2().tables[1].round(4)
# LASSO cross-check on standardised predictors.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
Xs = StandardScaler().fit_transform(X)
lasso = LassoCV(cv=5, random_state=RNG).fit(Xs, y)
pd.DataFrame({"LASSO_coef (standardised)": lasso.coef_}, index=DESCRIPTORS).round(3)

# 8. Principal Component Analysis
from sklearn.decomposition import PCA
scaler = StandardScaler().fit(X)
Z = scaler.transform(X)
pca = PCA().fit(Z)
scores = pca.transform(Z)

evr = pca.explained_variance_ratio_
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, len(evr)+1), evr); ax[0].set(xlabel="PC", ylabel="explained variance ratio",
                                                title="Scree")
ax[1].plot(range(1, len(evr)+1), np.cumsum(evr), "o-"); ax[1].axhline(0.8, color="r", ls="--")
ax[1].set(xlabel="number of PCs", ylabel="cumulative variance", title="Cumulative")
plt.tight_layout(); plt.show()
pd.DataFrame({"explained_var_ratio": evr.round(3),
              "cumulative": np.cumsum(evr).round(3)},
             index=[f"PC{i}" for i in range(1, len(evr)+1)])
loadings = pd.DataFrame(pca.components_.T, index=DESCRIPTORS,
                        columns=[f"PC{i}" for i in range(1, len(DESCRIPTORS)+1)])
loadings.iloc[:, :3].round(3)

# 9. Correlation circle
# correlation of each variable with PC1, PC2
corr_pc = loadings.iloc[:, :2].values * np.sqrt(pca.explained_variance_[:2])
fig, ax = plt.subplots(figsize=(7, 7))
circle = plt.Circle((0, 0), 1, fill=False, color="grey", ls="--")
ax.add_artist(circle)
for (x_, y_), name in zip(corr_pc, DESCRIPTORS):
    ax.arrow(0, 0, x_, y_, head_width=0.03, color="steelblue", length_includes_head=True)
    ax.text(x_*1.08, y_*1.08, name, fontsize=8, clip_on=False,
            ha="left" if x_ >= 0 else "right",
            va="bottom" if y_ >= 0 else "top")
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set(xlim=(-1.35, 1.35), ylim=(-1.25, 1.25),
       xlabel=f"PC1 ({evr[0]*100:.0f}%)", ylabel=f"PC2 ({evr[1]*100:.0f}%)",
       title="Correlation circle")
ax.set_aspect("equal"); plt.show()

# 10. Projection of the observations onto PC1-PC2
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(scores[:, 0], scores[:, 1], c=y, cmap="viridis", s=18, alpha=0.8)
plt.colorbar(sc, label="measured logS")
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set(xlabel=f"PC1 ({evr[0]*100:.0f}%) - polarity / H-bond capacity",
       ylabel=f"PC2 ({evr[1]*100:.0f}%) - aromatic-lipophilic vs aliphatic",
       title="Molecules projected on the first principal plane (colour = logS)")
plt.show()
print(f"corr(PC1, logS) = {np.corrcoef(scores[:,0], y)[0,1]:.3f}")
print(f"corr(PC2, logS) = {np.corrcoef(scores[:,1], y)[0,1]:.3f}")

# 11. Bonus - Principal Component Regression (PCR)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
r2_by_k = []
for k in range(1, len(DESCRIPTORS)+1):
    pcr = LinearRegression().fit(scores[:, :k], y)
    r2_by_k.append(r2_score(y, pcr.predict(scores[:, :k])))
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(DESCRIPTORS)+1), r2_by_k, "o-", label="PCR")
plt.axhline(ols.rsquared, color="r", ls="--", label=f"full OLS R²={ols.rsquared:.3f}")
plt.xlabel("number of principal components"); plt.ylabel("R²"); plt.legend()
plt.title("PCR: R² vs number of components"); plt.show()
pd.DataFrame({"PCR_R2": np.round(r2_by_k, 3)}, index=[f"first {k} PCs" for k in range(1, 9)])

# 12. Bonus - out-of-sample validation
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

Xsel, yv = X[chosen].values, y.values        # best-subset predictors from step 7
rows = []
for k, (tr, te) in enumerate(ShuffleSplit(n_splits=3, test_size=0.2, random_state=RNG).split(Xsel), 1):
    m = LinearRegression().fit(Xsel[tr], yv[tr])
    pred = m.predict(Xsel[te])
    rows.append({"split": k, "n_train": len(tr), "n_test": len(te),
                 "test_R2": r2_score(yv[te], pred),
                 "test_RMSE": mean_squared_error(yv[te], pred) ** 0.5})
res = pd.DataFrame(rows)
insample_r2 = LinearRegression().fit(Xsel, yv).score(Xsel, yv)
print(f"validated model: {len(chosen)} predictors {chosen}")
print(f"in-sample R² = {insample_r2:.3f}")
print(f"held-out R² = {res.test_R2.mean():.3f} ± {res.test_R2.std():.3f}"
      f"   |   RMSE = {res.test_RMSE.mean():.3f} logS units")
res.round(3)
