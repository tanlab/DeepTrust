#####################################################################################
# Deep Trust is modified and extention version of https://github.com/XifengGuo/DCEC #
#####################################################################################


from pyts.image import RecurrencePlot
import pandas as pd
from sklearn.metrics import silhouette_score
import scipy.stats
from sklearn import cluster, mixture
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from tensorflow.python.keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import metrics
#from ConvAE import CAE
from sklearn import cluster, mixture
import argparse
import os
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from pyts.image import gaf, GramianAngularField, recurrence, RecurrencePlot
from pyts.image import RecurrencePlot
from scipy.misc import imresize 

def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    return model

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepTrust(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0):

        super(DeepTrust, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])

    def pretrain(self, x, batch_size=128, epochs=300, optimizer='adam'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')

        # begin training
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=0)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=64, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None):

        save_interval = x.shape[0] / batch_size * 5

        # Step 1: pretrain if necessary
        if not self.pretrained and cae_weights is None:
            self.pretrain(x, batch_size)
            self.pretrained = True

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means or gmm.')
        """
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        """

        # You can also want to initialize the weights with Gaussian Mixture Model

        gmm = mixture.GaussianMixture(n_components=self.n_clusters, covariance_type="full")
        gmm = mixture.GaussianMixture(n_components=self.n_clusters, n_init=20)
        self.y_pred = gmm.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([gmm.means_])
        
        # Step 3: deep clustering
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1
            ite += 1



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='train')
    
    parser.add_argument('--dataset', default='simulated')
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--maxiter', default=420, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    args = parser.parse_args()

    # load dataset
    if args.dataset == 'simulated':
        x_vec = pd.read_pickle("data/simulated-data/simulated_timeseries.p")
        x = RecurrencePlot(percentage=20).fit_transform(x_vec)
        x = x.reshape(x.shape + (1,))
        y = pd.read_pickle("data/simulated-data/simulated_target.p")
        n_clusters = len(np.unique(y))
    elif args.dataset == 'biological':
        x_vec = pd.read_pickle ("data/biological-data/preprocessed/real_data_timeseries_new.p")
        x = pd.read_pickle ("data/biological-data/preprocessed/real_image_data_new.p")
        print(x_vec.shape)
        print(x.shape)
        x = x.reshape(x.shape + (1,))
        print(x.shape)
        x = x/255.0
        y = None
        n_clusters = 5
        #x = RecurrencePlot(percentage=20).fit_transform(x_vec)
        #################################################################################
        # You can transform biological timeseries dataset to the image with below code
        # Because we already did it, now we comment this part.
        # if you want to you use this part, you can:

        # pd.read_pickle ("data/biological-data/preprocessed/preprocessed_real_data.p")

        #data_deleted = x_vec.drop('Gene Symbol', 1)
        #data_matrix = np.asmatrix(data_deleted)
        #First element normalization
        #data_matrix = (data_matrix.T-data_matrix.T[0]).T
        #Get images
        #data_rp = RecurrencePlot(percentage=20).fit_transform(data_matrix)
        #Resize images to 28x28
        #x = np.empty(shape=(data_rp.shape[0],8,8))

        #for i in range(data_rp.shape[0]):
        #    x[i,:,:] = imresize(data_rp[i,],size=[8,8])
        #################################################################################
        
        

    if args.dataset == "simulated":

        kf = StratifiedKFold(n_splits=10)
        kf.get_n_splits(x)    


        my_counter = 1
        for train_index, test_index in kf.split(x, y):
            
            ## This split is for deepTrust
            x_train = x[train_index]
            x_test = x[test_index]


            y_train = y[train_index]
            y_test = y[test_index]

            ## This split is for baselines
            x_vec_train = x_vec[train_index]
            x_vec_test = x_vec[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            
            # Store all results in all_results dictionary
            all_results = {}

            deepTrust_model = DeepTrust(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=n_clusters)


            # begin clustering.
            print("================ DeepTrust is starting..=====================")
            optimizer = 'adam'
            deepTrust_model.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
            deepTrust_model.fit(x_train, y=y_train, tol=args.tol, maxiter=args.maxiter,
                     update_interval=args.update_interval,
                     cae_weights=args.cae_weights)
            print("================ Model training is finished..=====================")
            y_pred = deepTrust_model.y_pred
            

            #y_pred = q.argmax(1)
            y_pred = deepTrust_model.predict(x_test)
            
            temp = {}
            if args.dataset == 'simulated':
                temp['acc'] = metrics.acc(y_test, y_pred)
            else:
                temp['acc'] = -1
            temp['sil'] = silhouette_score(deepTrust_model.encoder.predict(x_test), y_pred)
            #temp['sil'] = silhouette_score(x_test, y_pred)
            all_results['deepTrust'] = temp
            
            del deepTrust_model
            print("================ Scores are calculated..=====================")


            ## Baselines
            print("================ Baseline models are starting..=====================")        
            number_of_clusters = n_clusters        
            ######################################## GMM ################################################
            gmm = mixture.GaussianMixture(n_components=number_of_clusters, covariance_type="full")
            gmm.fit(x_vec_train)
            centers = np.empty(shape=(gmm.n_components, x_vec_train.shape[1]))
            for i in range(gmm.n_components):
                density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(x_vec_train)
                centers[i, :] = x_vec_train[np.argmax(density)]
            
            pred = gmm.predict(x_vec_test)
            
            temp = {}
            if args.dataset == 'simulated':
                temp['acc'] = metrics.acc(y_test, pred)
            else:
                temp['acc'] = -1
            temp['sil'] = silhouette_score(x_vec_test, pred)
            all_results['gmm'] = temp

            ######################################## K Means ################################################
            kmeans = cluster.KMeans(n_clusters=number_of_clusters, n_init=20)
            kmeans.fit(x_vec_train)
            pred = kmeans.predict(x_vec_test)

            temp = {}
            if args.dataset == 'simulated':
                temp['acc'] = metrics.acc(y_test, pred)
            else:
                temp['acc'] = -1
            temp['sil'] = silhouette_score(x_vec_test, pred)
            all_results['kmeans'] = temp

            ######################################## Ward ################################################
            ward = cluster.AgglomerativeClustering(n_clusters=number_of_clusters, linkage='ward')
            pred = ward.fit_predict(x_vec_test)

            temp = {}
            if args.dataset == 'simulated':
                temp['acc'] = metrics.acc(y_test, pred)
            else:
                temp['acc'] = -1
            temp['sil'] = silhouette_score(x_vec_test, pred)
            all_results['ward'] = temp

            my_counter += 1
            pd.to_pickle(all_results, "all_results_"+args.dataset+"_"+str(my_counter)+".p")

    else: ## BIOLOGICAL

        # Store all results in all_results dictionary
        my_counter = 1
        all_results = {}

        deepTrust_model = DeepTrust(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=n_clusters)


        # begin clustering.
        print("================ DeepTrust is starting..=====================")
        optimizer = 'adam'
        deepTrust_model.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
        deepTrust_model.fit(x, y=None, tol=args.tol, maxiter=args.maxiter,
                 update_interval=args.update_interval,
                 cae_weights=args.cae_weights)
        print("================ Model training is finished..=====================")
        y_pred = deepTrust_model.y_pred
        

        #y_pred = q.argmax(1)
        y_pred = deepTrust_model.predict(x)
        
        temp = {}
        if args.dataset == 'simulated':
            temp['acc'] = metrics.acc(y_test, y_pred)
        else:
            temp['acc'] = -1
        temp['sil'] = silhouette_score(deepTrust_model.encoder.predict(x), y_pred)
        #temp['sil'] = silhouette_score(x_test, y_pred)
        all_results['deepTrust'] = temp
        print(all_results)
        del deepTrust_model
        print("================ Scores are calculated..=====================")


        ## Baselines
        print("================ Baseline models are starting..=====================")        
        number_of_clusters = n_clusters        
        ######################################## GMM ################################################
        gmm = mixture.GaussianMixture(n_components=number_of_clusters, covariance_type="full")
        gmm.fit(x_vec)
        centers = np.empty(shape=(gmm.n_components, x_vec.shape[1]))
        for i in range(gmm.n_components):
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(x_vec)
            centers[i, :] = x_vec[np.argmax(density)]
        
        pred = gmm.predict(x_vec)
        
        temp = {}
        if args.dataset == 'simulated':
            temp['acc'] = metrics.acc(y_test, pred)
        else:
            temp['acc'] = -1
        temp['sil'] = silhouette_score(x_vec, pred)
        all_results['gmm'] = temp
        print("gmm:", temp)
        ######################################## K Means ################################################
        kmeans = cluster.KMeans(n_clusters=number_of_clusters, n_init=20)
        kmeans.fit(x_vec)
        pred = kmeans.predict(x_vec)

        temp = {}
        if args.dataset == 'simulated':
            temp['acc'] = metrics.acc(y_test, pred)
        else:
            temp['acc'] = -1
        temp['sil'] = silhouette_score(x_vec, pred)
        all_results['kmeans'] = temp
        print("kmeans:", temp)
        ######################################## Ward ################################################
        ward = cluster.AgglomerativeClustering(n_clusters=number_of_clusters, linkage='ward')
        pred = ward.fit_predict(x_vec)

        temp = {}
        if args.dataset == 'simulated':
            temp['acc'] = metrics.acc(y_test, pred)
        else:
            temp['acc'] = -1
        temp['sil'] = silhouette_score(x_vec, pred)
        all_results['ward'] = temp
        print("ward:", temp)
        my_counter += 1
        pd.to_pickle(all_results, "all_results_"+args.dataset+"_"+str(my_counter)+".p")