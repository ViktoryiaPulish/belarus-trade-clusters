from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "trade_long.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

SEED = 42
YEAR = 2021
CONCENTRATION = 40
K_RANGE = range(2, 9)
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", font="DejaVu Sans")

SHORT = {
    "Электроника и ИКТ": "Электроника",
    "Машины и оборудование": "Машины",
    "Химия и полимеры": "Химия",
    "Транспорт": "Транспорт",
    "Фармацевтика": "Фарма",
    "Металлы и металлоизделия": "Металлы",
    "Продовольствие и сырьё для АПК": "Продовольствие",
    "Деревообработка и мебель": "Дерево/мебель",
}

REGIONS = {
    "СНГ": ["Россия", "Казахстан", "Украина", "Узбекистан", "Азербайджан",
            "Армения", "Кыргызстан", "Таджикистан", "Молдова"],
    "ЕС": ["Германия", "Польша", "Литва", "Латвия", "Италия", "Нидерланды",
           "Чехия", "Словакия", "Франция", "Великобритания", "Австрия",
           "Бельгия", "Испания"],
    "Дальнее зарубежье": ["Китай", "Турция", "США", "Бразилия", "Индия", "Япония"],
}

ISO3 = {
    "Россия": "RUS", "Казахстан": "KAZ", "Украина": "UKR", "Узбекистан": "UZB",
    "Азербайджан": "AZE", "Армения": "ARM", "Кыргызстан": "KGZ", "Таджикистан": "TJK",
    "Молдова": "MDA", "Германия": "DEU", "Польша": "POL", "Литва": "LTU", "Латвия": "LVA",
    "Италия": "ITA", "Нидерланды": "NLD", "Чехия": "CZE", "Словакия": "SVK",
    "Франция": "FRA", "Великобритания": "GBR", "Австрия": "AUT", "Бельгия": "BEL",
    "Испания": "ESP", "Китай": "CHN", "Турция": "TUR", "США": "USA", "Бразилия": "BRA",
    "Индия": "IND", "Япония": "JPN",
}

REGION_MULT = {
    "СНГ": {"Электроника и ИКТ": 0.7, "Машины и оборудование": 1.5, "Химия и полимеры": 0.8,
            "Транспорт": 1.4, "Фармацевтика": 1.2, "Металлы и металлоизделия": 0.9,
            "Продовольствие и сырьё для АПК": 1.8, "Деревообработка и мебель": 0.6},
    "ЕС": {"Электроника и ИКТ": 0.8, "Машины и оборудование": 1.0, "Химия и полимеры": 1.4,
           "Транспорт": 0.8, "Фармацевтика": 1.1, "Металлы и металлоизделия": 1.3,
           "Продовольствие и сырьё для АПК": 0.4, "Деревообработка и мебель": 2.0},
    "Дальнее зарубежье": {"Электроника и ИКТ": 2.0, "Машины и оборудование": 0.9,
                          "Химия и полимеры": 1.5, "Транспорт": 0.7, "Фармацевтика": 1.0,
                          "Металлы и металлоизделия": 1.1,
                          "Продовольствие и сырьё для АПК": 0.7, "Деревообработка и мебель": 0.5},
}

df = pd.read_csv(DATA, encoding="utf-8-sig")
df21 = df[df["Год"] == YEAR].copy()
categories = sorted(df21["Категория"].unique())
assert len(categories) == 8, f"Ожидалось 8 категорий, получено {len(categories)}"

base = pd.Series(1 / 8, index=categories)
base.rename("Базовый_вес").to_csv(OUT / "base_weights.csv", encoding="utf-8-sig")

rows = []
for region, countries in REGIONS.items():
    mult = pd.Series(REGION_MULT[region]).reindex(categories)
    w = (base * mult).values
    w = w / w.sum()
    for country in countries:
        shares = rng.dirichlet(w * CONCENTRATION)
        rows.append([country, region, ISO3[country], *shares])

X = pd.DataFrame(rows, columns=["Страна", "Регион", "ISO3", *categories])
X.to_csv(OUT / "country_shares.csv", index=False, encoding="utf-8-sig")
M = X.set_index("Страна")[categories]

scaler = StandardScaler()
Z = scaler.fit_transform(M.values)

sil = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
    sil[k] = silhouette_score(Z, km.labels_)
sil = pd.Series(sil, name="silhouette")
sil.rename_axis("k").to_csv(OUT / "silhouette_scores.csv", encoding="utf-8-sig")
best_k = int(sil.idxmax())

kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(Z)
X["Кластер"] = kmeans.labels_
X[["Страна", "Регион", "ISO3", "Кластер"]].to_csv(
    OUT / "clusters.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(6, 4))
plt.plot(sil.index, sil.values, marker="o")
plt.axvline(best_k, ls="--", c="red", label=f"выбрано k={best_k}")
plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title("Выбор числа кластеров")
plt.legend(); plt.tight_layout(); plt.savefig(OUT / "silhouette.png", dpi=150); plt.close()

profiles = X.groupby("Кластер")[categories].mean() * 100
overall = M.mean() * 100
deviation = profiles - overall
profiles.round(1).to_csv(OUT / "cluster_profiles.csv", encoding="utf-8-sig")
deviation.round(1).to_csv(OUT / "cluster_deviation_pp.csv", encoding="utf-8-sig")
crosstab = pd.crosstab(X["Кластер"], X["Регион"])
crosstab.to_csv(OUT / "cluster_vs_region.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * best_k + 3))
sns.heatmap(profiles.rename(columns=SHORT), annot=True, fmt=".1f", cmap="YlGnBu",
            cbar_kws={"label": "% оборота"}, ax=axes[0])
axes[0].set_title("Профили кластеров: средняя доля категории, %")
sns.heatmap(deviation.rename(columns=SHORT), annot=True, fmt="+.1f", cmap="RdBu_r",
            center=0, cbar_kws={"label": "п.п."}, ax=axes[1])
axes[1].set_title("Отклонение от среднего по всем странам, п.п.")
for ax in axes:
    ax.set_xlabel(""); ax.set_ylabel("Кластер")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
plt.tight_layout(); plt.savefig(OUT / "heatmap_profiles.png", dpi=150); plt.close()

pca = PCA(n_components=2, random_state=SEED)
P = pca.fit_transform(Z)
X["PC1"], X["PC2"] = P[:, 0], P[:, 1]
markers = {"СНГ": "o", "ЕС": "s", "Дальнее зарубежье": "^"}
palette = sns.color_palette("tab10", best_k)

plt.figure(figsize=(9, 7))
for region, mk in markers.items():
    sub = X[X["Регион"] == region]
    plt.scatter(sub["PC1"], sub["PC2"], marker=mk, s=120,
                c=[palette[c] for c in sub["Кластер"]], edgecolor="k", label=region)
for _, r in X.iterrows():
    plt.annotate(r["Страна"], (r["PC1"], r["PC2"]), xytext=(4, 4),
                 textcoords="offset points", fontsize=8)
ev = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({ev[0]:.1f}%)"); plt.ylabel(f"PC2 ({ev[1]:.1f}%)")
plt.title(f"PCA стран по структуре торговли (цвет — кластер, k={best_k}; форма — регион)")
plt.legend(title="Регион"); plt.tight_layout()
plt.savefig(OUT / "pca_scatter.png", dpi=150); plt.close()

loadings = pd.DataFrame(pca.components_.T, index=[SHORT[c] for c in categories],
                        columns=["PC1", "PC2"])
loadings.round(3).to_csv(OUT / "pca_loadings.csv", encoding="utf-8-sig")

X["Кластер_"] = "Кластер " + X["Кластер"].astype(str)
X["Топ-3"] = M.apply(lambda s: ", ".join(
    f"{SHORT[c]} {v*100:.0f}%" for c, v in s.sort_values(ascending=False).head(3).items()),
    axis=1).values
fig = px.choropleth(
    X, locations="ISO3", color="Кластер_", hover_name="Страна",
    hover_data={"Регион": True, "Топ-3": True, "ISO3": False, "Кластер_": False},
    category_orders={"Кластер_": sorted(X["Кластер_"].unique())},
    color_discrete_sequence=px.colors.qualitative.Set1,
    title=f"Кластеры стран по структуре торговли с Беларусью (K-means, k={best_k})")
fig.add_trace(go.Scattergeo(lon=[27.56], lat=[53.9], text=["Беларусь"], mode="markers+text",
                            marker=dict(size=10, color="black", symbol="star"),
                            textposition="top center", name="Беларусь"))
fig.update_geos(showcountries=True, countrycolor="lightgray", projection_type="natural earth")
fig.write_html(OUT / "map_clusters.html")
try:
    fig.write_image(OUT / "map_clusters.png", width=1400, height=800, scale=2)
except Exception as e:
    print(f"[warn] PNG карты не сохранён: {e}")

print(f"k={best_k}, silhouette={sil[best_k]:.3f}")
print(crosstab)
print(f"Результаты: {OUT}")