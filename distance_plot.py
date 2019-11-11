import argparse
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets.samples_generator import make_blobs
from dataloader import minist_loader
from dataloader import cifar_loader
import t3nsor as t3
from my_models import LinearSVM
import easydict as edict
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from numpy import linalg as LA

from sklearn.metrics import accuracy_score

# tensor = tl.tensor(np.arange(210*2).reshape((5, 6, 7, 2)), dtype=tl.float64)
# tucker_tensor = tucker(tensor, rank=[5, 5, 2, 2])
#data structure of tucker decomposition:
#tucker_tensor is a (core_tensor_tuple, factor_tuple)
#core_tensor_tuple = (first slice, next slice,...)
#factor_tuple = (factor1, factor2,... )


settings = edict.EasyDict({
    "GPU": False,
    "BATCH_SIZE": 1000,
    "WORKERS": 12,
    #"TT_SHAPE": [32, 32],
    #"TT_SHAPE": [4, 8, 4, 8],
    "TT_SHAPE": [2,2,2,2,2,2,2,2,2,2],
    #"TUCKER_RANK": [16, 16],
    #"TUCKER_RANK": [4, 4, 4, 4],
    "TUCKER_RANK": [2,2,2,2,2,2,2,2,2,2],
    "TT_RANK": 16,
    "UNIFORM": False,
    "NORMAL": True,
    "INDEP_SAMPLE": False,
    "FEATURE_EXTRACT": True,
    "MINIST_SHAPE": [28, 28],
    "CIFAR_SHAPE": [32, 32],
    })

# val_loader = minist_loader(settings, 'val')
# train_loader = minist_loader(settings, 'train', shuffle=True)
# dataiter = iter(train_loader)
# input,target = dataiter.next()
# print('MINIST data observation:', input[0, 0, :, :])

val_loader = cifar_loader(settings, 'val')
train_loader = cifar_loader(settings, 'train', shuffle=True)
dataiter = iter(train_loader)
input, target = dataiter.next()
print('CIFAR data observation:', input[0, 0, :, :])

if settings.INDEP_SAMPLE:
    input = torch.rand(settings.BATCH_SIZE, 1, 32, 32)
    if settings.NORMAL:
        for batch_idx in range(0, settings.BATCH_SIZE):
            input[batch_idx, 0, :, :] = torch.randn(32, 32)
    if settings.UNIFORM:
        for batch_idx in range(0, settings.BATCH_SIZE):
            input[batch_idx, 0, :, :] = 30*torch.rand(32, 32)
            #print('artifical data:', input[batch_idx, 0, :, :])
            #pdb.set_trace()

else:
    if settings.UNIFORM:
        input = torch.randn(settings.BATCH_SIZE, 1, 32, 32)
    if settings.NORMAL:
        input = torch.randn(settings.BATCH_SIZE, 1, 32, 32)
#pdb.set_trace()


#quantitative measure of level of diagonalization of a matrix:
def output_level_diag(tensor):
    ndims = len(tensor.shape)
    if ndims == 2:
        vector = np.reshape(tensor, (1, -1))
        diag_level = np.sum(np.absolute(np.diag(tensor)))/(np.sum(np.absolute(vector)) - np.sum(np.absolute(np.diag(tensor))))
        print('level of diagnal:', diag_level)
        return diag_level
        #pdb.set_trace()


#input is a batch of MINIST data
#input shape: [batch_size, 1, 28, 28]
def input_to_tt_tensor(input, settings):
    #input shape: batch_size, num_channels, size_1, size_2
    #convert input from dense format to TT format, specificaly, TensorTrainBatch
    #input shape: [batch_size 1 28 28]
    data_list = []
    core_1_diag_level_list = []
    core_2_diag_level_list = []
    for batch_iter in range(settings.BATCH_SIZE):

        tt_cores_curr = t3.to_tt_tensor(input[batch_iter, 0, :, :].contiguous().view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK).tt_cores

        #test reconstruction
        tt = t3.to_tt_tensor(input[batch_iter, 0, :, :].contiguous().view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK)

        print('reconstruction error:', torch.norm(tt.full().view(32, 32)-input[batch_iter, 0, :, :]))
        if len(tt_cores_curr) == 2:
            core_1_diag_level_list.append(output_level_diag(np.reshape(tt_cores_curr[0].numpy(), (32, -1))))
            core_2_diag_level_list.append(output_level_diag(np.reshape(tt_cores_curr[1].numpy(), (32, -1))))
        i = 0
        for tt_core in tt_cores_curr:
            print('batch_iter:', batch_iter, i, 'th tt_core', 'shape:', tt_core.shape, tt_core)
            if i == 0:
                data_vector = torch.reshape(tt_core, shape=[1, -1])
            else:
                data_vector = torch.cat((data_vector, torch.reshape(tt_core, shape=[1, -1])), dim=1) #shape: [1,n]
            i += 1
        data_list.append(data_vector)
        #pdb.set_trace()
    if len(settings.TT_SHAPE) == 2:
        mean_diag_level_core1 = sum(core_1_diag_level_list) / len(core_1_diag_level_list)
        mean_diag_level_core2 = sum(core_2_diag_level_list) / len(core_2_diag_level_list)
        print('mean_diag_level_core1:', mean_diag_level_core1, 'mean_diag_level_core2:', mean_diag_level_core2)
    data_2d = torch.cat(data_list, dim=0)
    print('data_2d shape:', data_2d.shape)
    assert(data_2d.shape[0] == settings.BATCH_SIZE)
    return data_2d.numpy()


def input_to_svd(input, settings):
    data_list = []
    for batch_iter in range(settings.BATCH_SIZE):
        u, s, v = torch.svd(input[batch_iter, 0, :, :])
        uu, ss, vv = torch.reshape(u, [1, -1]), torch.reshape(s, [1, -1]), torch.reshape(v, [1, -1])
        data_vector = torch.cat((uu, ss, vv), dim=1)
        assert(data_vector.shape[0] == 1)
        data_list.append(data_vector)
    data_2d = torch.cat(data_list, dim=0)
    assert(data_2d.shape[0] == settings.BATCH_SIZE)
    return data_2d.numpy()


def input_to_tucker(input, settings):
    data_list = []
    diag_level = []
    input = input.numpy()
    for batch_idx in range(0, settings.BATCH_SIZE):
        tucker_tensor_core, tucker_tensor_factor = tucker(np.reshape(input[batch_idx, 0, :, :], settings.TT_SHAPE), rank=settings.TUCKER_RANK)
        #output_max_position(tucker_tensor_core)
        print('tucker_tensor_core:', tucker_tensor_core)
        diag_level.append(output_level_diag(tucker_tensor_core))
        #pdb.set_trace()
        #assert(tucker_tensor_core.shape == settings.TUCKER_RANK)
        #print('reconstruction error:', LA.norm(np.reshape(tl.tucker_to_tensor(core=tucker_tensor_core, factors=tucker_tensor_factor), (28, 28)) - input[batch_idx, 0, :, :]))
        core_vector = np.reshape(tucker_tensor_core, (1, -1))
        #pdb.set_trace()
        j = 0
        for factor_slice in tucker_tensor_factor:
            if j == 0:
                factor_vector = np.reshape(factor_slice, (1, -1))
            else:
                #pdb.set_trace()
                factor_vector = np.concatenate((factor_vector, np.reshape(factor_slice, (1, -1))), axis=1)
                assert(factor_vector.shape[0] == 1)
            j += 1
        all_vector = np.concatenate((core_vector, factor_vector), axis=1)
        data_list.append(all_vector)
    if len(settings.TT_SHAPE) == 2:
        mean_diag_level = sum(diag_level)/len(diag_level)
        print('mean_diag_level:', mean_diag_level)

    data_2d = np.concatenate(data_list, axis=0)
    assert(data_2d.shape[0] == settings.BATCH_SIZE)
    return data_2d


#input_tt = input_to_tucker(input, settings)
#input_tt = input_to_svd(input, settings)
input_tt = input_to_tt_tensor(input, settings)


distort = torch.randn(settings.BATCH_SIZE, 1)
for i in range(0, settings.BATCH_SIZE):
    random_int = torch.randint(0, settings.BATCH_SIZE, (2,))
    #print('random_index pair:', random_int)
    distort[i, 0] = torch.norm(torch.Tensor(input_tt[random_int[0], :] - input_tt[random_int[1], :]))/torch.norm(input[random_int[0], 0, :, :] - input[random_int[1], 0, :, :])
    #print(i, 'th distort:', distort[i, 0])

x_axis = list(range(0, settings.BATCH_SIZE))
x_axis = torch.Tensor(x_axis).view(settings.BATCH_SIZE, -1)
#print(x_axis)

plt.scatter(x_axis, distort, marker='.')

plt.ylim(0, 2)
plt.clim(-0.5, 9.5)
plt.show()

print('input shape:', input_tt.shape)
target = target.numpy()
X_train, X_test, y_train, y_test = train_test_split(input_tt, target, test_size=0.25, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_test = sc_X.transform(test)

print('SVM Classifier with gamma = 0.1; Kernel = Polynomial')
classifier = SVC(gamma='auto', kernel='sigmoid', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

model_acc = classifier.score(X_test, y_test)


print('\nSVM Trained Classifier Accuracy: ', model_acc)

