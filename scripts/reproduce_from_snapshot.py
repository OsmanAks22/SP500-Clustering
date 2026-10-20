"""Re-run the clustering on the saved metric table, without downloading anything.

Usage: python scripts/reproduce_from_snapshot.py [path/to/metrics.csv]

Applies the notebook's preparation (1st/99th percentile winsorising, then
StandardScaler) and fits Ward agglomerative clustering and K-Means with K = 2,
then prints the silhouettes, cluster sizes, the cross-tabulation and the share
of stocks on which the two methods agree.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Mean_Daily_Return", "Beta", "Annual_Volatility"]
DEFAULT = Path(__file__).resolve().parent.parent / "data" / "sp500_metrics_2022_2024.csv"


def main(path: Path) -> None:
    metrics = pd.read_csv(path, index_col=0)
    X = metrics[FEATURES].copy()
    for col in FEATURES:
        lo, hi = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(lo, hi)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X_scaled)
    agg = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X_scaled)

    print(f"Stocks: {len(metrics)}")
    print(f"Silhouette  K-Means: {silhouette_score(X_scaled, km):.4f}   "
          f"Ward: {silhouette_score(X_scaled, agg):.4f}")
    print(f"Sizes       K-Means: {sorted(np.bincount(km).tolist())}   "
          f"Ward: {sorted(np.bincount(agg).tolist())}")
    table = pd.crosstab(pd.Series(km, name="K-Means"), pd.Series(agg, name="Ward"))
    print(table)
    # Cluster ids are arbitrary, so take the better of the two possible matchings.
    same = max(np.trace(table.values), table.values[0, 1] + table.values[1, 0])
    print(f"Agreement: {same}/{len(metrics)} = {same / len(metrics):.1%}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
