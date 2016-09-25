import tensorflow as tf
import numpy as np

from functions import *

n_features = 2					# Number of features, here x and y i.e. 2
n_clusters = 3					# Number of clusters to create and identify
n_samples_per_cluster = 1000	# Number of samples in each cluster
seed = 700						# Used for cluster randomization
embiggen_factor = 50
n_iterations = 100				# Number of maximum iterations to train
epsilon = 0.0001				# Epsilon - threshold for stopping

# Create Randomized clusters
data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
# Create starting point for training
initial_centroids = choose_random_centroids(samples, n_clusters)
# Inintialize centroids to plot
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

# Plot Initial centroids
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)
plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)

# Trian the model
model = tf.initialize_all_variables()
with tf.Session() as session:
    i=0
    while i<n_iterations and not session.run(tf.less(tf.abs(tf.sub(updated_centroids, tf.to_float(data_centroids) )), epsilon)).all():
    	if i%(n_iterations/10) == 0:
            print("Step : "+str(i))
        nearest_indices = assign_to_nearest(samples, updated_centroids)
        updated_centroids = update_centroids(samples, nearest_indices, n_clusters)
        i = i+1
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)

# Plot predicted centroids
plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)