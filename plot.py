import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import plotly
plotly.tools.set_credentials_file(username='%username%', api_key='%api_key%')

scatter_data, cluster_data, scatter_anomalies = [], [], []
text_data, text_anomalies = [], []	# индекс выбранной точки в массиве data + anomalies

def create_plot(data, anomalies):
	X = np.array(data + anomalies)
	Y = np.array([1] * len(data) + [0] * len(anomalies))

	algorithm = IsolationForest(contamination=0.01, n_jobs=cpu_count())
	algorithm.fit(data)
	pred = algorithm.predict(data + anomalies)
 
	for i in list(set(np.random.randint(0, len(X), size=1000))):
	    scatter_data.append(X[i]), text_data.append(str(i)) if Y[i] == 1 else scatter_anomalies.append(X[i]), text_anomalies.append(str(i))
	    if pred[i] == 1:
		cluster_data.append(X[i])

	pca_3d = PCA(n_components=3)
	pca_3d.fit(txt_w2v)
	scatter_data_3d = pca_3d.transform(scatter_data)
	cluster_data_3d = pca_3d.transform(cluster_data)
	scatter_anomalies_3d = pca_3d.transform(scatter_anomalies)

	scatter_data = dict(mode = 'markers',
			    type = 'scatter3d',
			    name = 'banki.ru',
			    text = text_banki,
			    x = scatter_data_3d[:, 0],
			    y = scatter_data_3d[:, 1],
			    z = scatter_data_3d[:, 2],
			    marker = dict(size=2, color='rgb(23, 190, 207)'))

	clusters_data = dict(alphahull = 1,
			     opacity = 0.1,
			     type = "mesh3d",
			     name = 'cluster banki.ru',
			     x = cluster_data_3d[:, 0],
			     y = cluster_data_3d[:, 1],
			     z = cluster_data_3d[:, 2])

	scatter_anomalies = dict(mode = 'markers',
				 type = 'scatter3d',
				 name = 'lenta.ru',
				 text = text_lenta,
				 x = scatter_anomalies_3d[:, 0],
				 y = scatter_anomalies_3d[:, 1],
				 z = scatter_anomalies_3d[:, 2],
				 marker = dict(size=2, color='rgb(255, 37, 25)'))

	fig = dict(data=[scatter_data, clusters_data, scatter_anomalies])

	plotly.plotly.iplot(fig, filename='3d point clustering')
