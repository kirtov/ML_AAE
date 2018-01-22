import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_path", type=str)
args = parser.parse_args()

%matplotlib inline
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import _pickle as pickle
import sys
import json
import math
import time
from keras.models import model_from_json
import copy
import gc
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import Adadelta, Adagrad, Adam, RMSprop, Nadam
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape,Input, Dense
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
import h5py
import keras.callbacks as ckbs
from keras.models import Model
from keras.layers import concatenate as concat
from copy import deepcopy
from sklearn.metrics import log_loss
from sklearn.grid_search import ParameterGrid
np.random.seed(5531)
import os
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')

def grid_generator(search_space):
    param_grid = ParameterGrid(search_space)
    all_params = []
    for p in param_grid:
        all_params.append(p)
    for key in search_space.keys():
        if (isinstance(search_space[key], dict)):
            new_params=[]
            for param in all_params:
                if (search_space[key][param[key]] is None):
                    new_params.append(param)
                else:
                    param_grid = ParameterGrid(search_space[key][param[key]])
                    add_params = [p for p in param_grid]
                    for aparam in add_params:
                        tparam = copy.copy(param)
                        tparam.update(aparam)
                        new_params.append(tparam)
            all_params = new_params
    for param in all_params:
        yield param

def get_optimizer(name, lr):
    if (name == 'adam'):
        return Adam(lr = lr)
    elif (name == 'adadelta'):
        return Adadelta(lr = lr)
    elif (name == 'adagrad'):
        return Adagrad(lr = lr)
    elif (name == 'rmsprop'):
        return RMSprop(lr = lr)
    elif (name == 'nadam'):
        return Nadam(lr = lr)
    elif (name == 'sgd'):
        return SGD(lr = lr)

def get_activation(act):
    str_act = ['relu', 'tanh', 'sigmoid', 'linear','softmax','softplus','softsign','hard_sigmoid']
    if (act in str_act):
        return Activation(act)
    else:
        return {'prelu': PReLU(), 'elu' : ELU(), 'lrelu' : LeakyReLU(),
               }[act]


class AdversarialAutoencoder():
    def __init__(self, exp_dir, input_shape1, params):
        self.shuffle = True
        self.input_shape1 = input_shape1
        optimizer = get_optimizer(params['optimizer'], params['lr'])
        self.exp_dir = exp_dir
        self.ae, self.encoder, dec1 = self.build_ae(input_shape1, params)
        self.ae.compile(loss='mse', optimizer=optimizer)
        self.encoder.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.discriminator = self.build_discriminator(params)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=get_optimizer(params['optimizer'], params['lr']), metrics=['accuracy'])
        self.generator_enc, inps = self.build_generator(params)
        gan_inputs = []
        for i in inps:
            gan_inputs.append(Input(shape=(self.p['gen_inp_dim'],)))
        self.discriminator.trainable = False
        self.generator_enc.compile(loss='mse', optimizer=optimizer)
        in1 = Input((params['latent_dim'],))
        self.generator_dec = Model(in1, dec1(in1))
        self.generator_dec.compile(loss = 'binary_crossentropy', optimizer=optimizer)
        gen = self.generator_enc(gan_inputs)
        self.generator = Model(gan_inputs, self.discriminator(gen))
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics =['accuracy'])
        
    def build_ae(self, is1, params):
        encoder, x1, inp1 = self.get_encoder(is1, params)
        encoder = (Dense(params['latent_dim'])(encoder))
        self.encoded_dim = K.int_shape(encoder)[-1]
        
        decoder1 = self.get_decoder(K.int_shape(x1)[1:], params, is1)

        gen_img1 = decoder1(encoder)

        return Model(inp1, gen_img1), Model(inp1, encoder), decoder1
    
    def get_encoder(self, im_shape, params):
        input_img1 = Input(shape=im_shape)
        x1 = Flatten()(input_img1)
        x1 = LeakyReLU(alpha=0.2)(Dense(512)(x1))
        x1 = LeakyReLU(alpha=0.2)(Dense(256)(x1))
        encoder1 = x1
        return encoder1, x1, input_img1
    
    def get_decoder(self, reshape_size, params, ins):
        decoder = Sequential()
        decoder.add(Dense(256, input_dim=self.encoded_dim))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(Dense(512))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(Dense(784, activation='tanh'))
        decoder.add(Reshape(self.input_shape1))
        return decoder
    
    def build_generator(self, params):
        inps = [Input((params['gen_inp_dim'],))]
        gen = Dense(params['gen_layers'][0])(inps[-1])
        gen = get_activation(params['activation'])(gen)
        for i, l in enumerate(params['gen_layers'][1:]):
            if (i + 2 in params['noise_layers']):
                inps.append(Input((params['gen_inp_dim'],)))
                gen = concat([gen, inps[-1]])
            gen = Dense(l)(gen)
            gen = get_activation(params['activation'])(gen)
        gen = get_activation(params['output_activation'])(Dense(params['latent_dim'])(gen))
        gen = Model(inps, gen)
        return gen, inps

    def build_discriminator(self, params):
        model = Sequential()
        model.add(Dense(512, input_dim=self.encoded_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    def pretrain(self, X_train, X_test, batch_size, patience = 150):
        wait = 0
        count_of_batches = int(X_train.shape[0] / batch_size)
        while (wait < patience):
            for i in range(count_of_batches):
                batch1 = X_train[i*batch_size:(i+1)*batch_size]
                self.ae.train_on_batch(batch1, batch1)
            wait += 1
        print("PRETRAIN", ae_loss)

    def fit(self, X_train, X_test):
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        X_train = X_train
        half_batch = int(batch_size / 2)
        count_of_batches = int(X_train.shape[0] / batch_size)
        print(count_of_batches)
        self.losses = []
        self.best_loss = 999
        self.best_s = 0
        self.best_corr_gen = 0
        self.wait = 0
        ae_loss = 1
        d_loss=[0,0]
        g_loss=[0,0]
        self.best_ae_weights = None
        train_discr = True
        self.pretrain(X_train, X_test, batch_size)
        for epoch in range(epochs):
            iters = 0
            while (d_loss[1] < 0.75 and iters < 10):
                iters += 1
                for i in range(count_of_batches):
                    batch1 = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
                    latent_real = self.encoder.predict(batch1)
                    latent_fake = self.generator_enc.predict(self.get_noise(batch_size))
                    self.discriminator.train_on_batch(latent_real, np.array([0.9] * len(batch1)))
                    self.discriminator.train_on_batch(latent_fake, np.zeros((batch_size, 1)))
                d_loss, _ = self.evaluate_disc(X_test)
            dw = copy.deepcopy(self.discriminator.get_weights())
            for i in range(count_of_batches):
                valid_y = np.ones((batch_size, 1))
                latent = self.get_noise(batch_size)
                self.generator.train_on_batch(latent, valid_y)
                self.discriminator.set_weights(dw)
            d_loss, g_loss, ae_loss, predicts = self.evaluate(X_test)
            self.losses.append([d_loss, g_loss, ae_loss])
            print(epoch, ae_loss, d_loss, g_loss)
            self.visualize_gen(epoch, self.params['gen_inp_dim'])
        return self.losses
    
    def visulize_gen(self, epoch, gen_dim, examples=25):
        noise = self.get_noise(examples)
        gens = self.generator_dec.predict(self.generator_enc.predict(noise))
        pickle.dump(gens, open(self.exp_dir + "/gen_images/{}_epoch.pkl".format(epoch), 'wb'))
        
    def get_noise(self, c):
        noise = []
        if (self.params['noise_diff']):
            for i in range(len(self.params['noise_layers'])):
                noise.append(np.random.normal(size=[c, self.params['gen_inp_dim']]))
        else:
            noise = [np.random.normal(size=[c, self.params['gen_inp_dim']])] * len(self.params['noise_layers'])
        return noise
        
    def evaluate_disc(self, X_test, noise = None):
        latent_valid = self.encoder.predict(X_test)
        if (noise is None):
            latent_noise = self.get_noise(X_test.shape[0])
        else:
            latent_noise = noise
        latent_fake = self.generator_enc.predict(latent_noise)
        valid = np.ones((X_test.shape[0], 1))
        fake = np.zeros((X_test.shape[0], 1))
        latent = np.concatenate((latent_fake, latent_valid))
        valid = np.concatenate((fake, valid))
        d_loss = self.discriminator.evaluate(latent, valid, verbose=0)
        return d_loss, latent_noise
    
    def evaluate_gen(self, noise = None):
        if (noise is None):
            latent_noise = self.get_noise(2000)
        else:
            latent_noise = noise
        return self.generator.evaluate(latent_noise, np.zeros((2000, 1)), verbose = 0)

    def evaluate(self, X_test):
        predicts = self.ae.predict(X_test)
        ae_loss = self.ae.evaluate(X_test, X_test, verbose=0)
        d_loss, latent_noise = self.evaluate_disc(X_test)
        g_loss = self.generator.evaluate(latent_noise, np.ones((X_test.shape[0], 1)), verbose = 0)
        return d_loss, g_loss, ae_loss, predicts
    
def add_experiment(X_train, X_test, exp_dir, params):
    aae = Adversarial_autoencoder(exp_dir, (1,28,28), params)
    losses = aae.fit(X_train, X_test)
    return aae, losses

def save_model(nn, path, prefix):
    pickle.dump([nn.to_json(), nn.get_weights()], open(path + "{}.pkl".format(prefix), 'wb'))
    
def save_aae(aae, path):
    save_model(aae.generator_enc, path, 'genenc')
    save_model(aae.generator_dec, path, 'gendec')
    save_model(aae.discriminator, path, 'disc')

def cnn_gridsearch(X_train, X_test, experiment_path, search_space):
    if (not os.path.exists(experiment_path)):
        os.mkdir(experiment_path)
    stats = pd.DataFrame(columns=['Optimizer', 'LR', 'Activation', 'Dropout', 'Disc_layers', 'Gen_layers',
                                 'Gen_input_dim', 'Latent dim', 'Noise layers', 'Different noise', 'Output activation'])
    history_filename = experiment_path + 'hist_cnn.h'
    grid = grid_generator(search_space)
    stats_filename = experiment_path + 'stats.csv'
    histories = {}
    i = 1
    for params in grid:
        print(i)
        cur_exp_path = experiment_path + "/{}_model/".format(i)
        if (not os.path.exists(cur_exp_path)):
            os.mkdir(cur_exp_path)
            os.mkdir(cur_exp_path + "/gen_images/")
        aae, losses = add_experiment(X_train, X_test, cur_exp_path, params)
        stats.loc[stats.shape[0]] = [params['optimizer'], params['lr'], params['activation'], 
                                    params['dropout'], params['disc_layers'], params['gen_layers'], 
                                    params['gen_inp_dim'], params['latent_dim'], params['noise_layers'], params['noise_diff'],
                                    params['output_activation']]
        save_aae(aae, cur_exp_path)
        histories[i] = losses
        pickle.dump(histories, open(history_filename, 'wb'))
        stats.to_csv(stats_filename, index=False)
        i += 1
        
search_space = { 'optimizer' : {'adam' : {'lr' : [0.001, 0.0002]}},
                 'activation' : ['lrelu'],
                 'batch_size' : [256],
                 'dropout' : [0.3],
                 'epochs' : [1500],
                 'gen_inp_dim' : [100,200],
                 'output_activation' : ['tanh'],
                 'gen_layers' : [[100,200,300], [200,400,500], [300,500,700]],
                 'disc_layers' : [[500,500,500]],
                 'noise_layers' : [[1], [1,2], [1,2,3]],
                 'noise_diff' : [True, False],
                 'latent_dim' : [100,200],
                   }

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5)/127.5

cnn_gridsearch(X_train, X_test, args.exp_path, search_space)
