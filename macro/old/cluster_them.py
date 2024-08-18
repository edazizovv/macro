
# [tbd]: double check with the factors that are considered by FRS
# gdp
# inflation
# employment
# fed rate
# real fed rate (fed rate subtracted inflation)
# real gdp (gdp subtracted inflation)
# pure gdp (gdp subtracted inflation and fed rate)
# long term uptrend in snp 500
# long term uptrend in snp 500 subtracted inflation
# long term uptrend in snp 500 subtracted inflation and fed rate

# cluster them and see what happens

# GDP__pct
# PCE
# UNRATE
# FEDFUNDS
# (1 + FEDFUNDS) / (1 + PCE) - 1
# (1 + GDP__pct) / (1 + PCE) - 1
# (1 + GDP__pct) / ((1 + PCE) * (1 + FEDFUNDS)) - 1
# at the moment we don't use snps

# PCEC96

import pandas
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from hdbscan import HDBSCAN
import seaborn
from matplotlib import pyplot

data = pandas.read_csv('./result.csv')
data = data.set_index('jx')

data['gdp'] = data['GDP__pct'].copy()
data['inf'] = data['PCE__pct'].copy()
data['emp'] = data['UNRATE'] / 100
data['fed'] = data['FEDFUNDS'] / 100
data['real_fed'] = ((1 + data['fed']) / (1 + data['inf'])) - 1
data['real_gdp'] = ((1 + data['gdp']) / (1 + data['inf'])) - 1
data['pure_gdp'] = ((1 + data['gdp']) / ((1 + data['inf']) * (1 + data['fed']))) - 1
data['ivv'] = data['IVV_MEAN__pct']
data['tlt'] = data['TLT_MEAN__pct']
data['pct_fed'] = data['FEDFUNDS__pct'].copy()
data['pct_emp'] = data['UNRATE__pct'].copy()

# x = data[['gdp', 'inf', 'emp', 'fed', 'real_fed', 'real_gdp', 'pure_gdp', 'pct_fed', 'pct_emp']].values
x = data[['gdp', 'inf', 'emp', 'real_gdp', 'pure_gdp', 'pct_fed', 'pct_emp']].values
# x = data[['gdp', 'inf', 'emp', 'real_gdp']].values
# x = data[['gdp', 'inf', 'emp']].values

scaler = StandardScaler()
x_whitened = scaler.fit_transform(X=x)

reducer5 = UMAP(n_components=5, densmap=True)          # dens_lambda=None
reducer2 = UMAP(n_components=2, densmap=True)

x_d5 = reducer5.fit_transform(X=x_whitened)
# x_d2 = reducer2.fit_transform(X=x_whitened)
x_d2 = reducer2.fit_transform(X=x_d5)

clusterer = HDBSCAN(min_cluster_size=10)    # min_cluster_size=None

clusters = clusterer.fit_predict(X=x_d5)
# pandas.Series(clusters).value_counts()

seaborn.scatterplot(x=x_d2[:, 0], y=x_d2[:, 1], hue=clusters)

cp = seaborn.color_palette("rocket", as_cmap=True)
# seaborn.scatterplot(x=x_d2[:, 0], y=x_d2[:, 1], hue=data['tlt'].values, palette=cp)
# pyplot.scatter(x_d2[:, 0], x_d2[:, 1], c=data['tlt'].values, s=5)

data['cluster'] = clusters
data_gb = data[['gdp', 'inf', 'emp', 'fed', 'real_fed', 'real_gdp', 'pure_gdp', 'ivv', 'tlt', 'pct_fed', 'pct_emp', 'cluster']].groupby(by='cluster').describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T # .unstack(1)

# seaborn.histplot(data=data, x='', hue='cluster', bins=20)


# FOUND: fed, real_fed, pure_gdp -- no targets



def target_defined(y):
    if y < -0.015:
        return -1
    elif y < 0.015:
        return 0
    else:
        return 1

data['target'] = data['TLT_MEAN__pct'].apply(func=target_defined)

embedder_d5 = UMAP(n_components=5, densmap=True)
embedder_d2 = UMAP(n_components=2, densmap=True)
embedding_d5 = embedder_d5.fit_transform(x, y=data['target'].values)
embedding_d2 = embedder_d2.fit_transform(embedding_d5, y=data['target'].values)

clusterer_emb = HDBSCAN(min_cluster_size=10)    # min_cluster_size=None

clusters_emb = clusterer_emb.fit_predict(X=embedding_d5)

# seaborn.scatterplot(x=embedding_d2[:, 0], y=embedding_d2[:, 1], hue=clusters_emb)
# seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=data['tlt'].values, palette=cp)
# pyplot.scatter(embedding[:, 0], embedding[:, 1], c=data['tlt'].values, s=5)

data['cluster_emb'] = clusters_emb
data_cb = data[['gdp', 'inf', 'emp', 'fed', 'real_fed', 'real_gdp', 'pure_gdp', 'ivv', 'tlt', 'pct_fed', 'pct_emp', 'cluster_emb']].groupby(by='cluster_emb').describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T # .unstack(1)
