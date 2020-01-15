from scipy.stats import entropy
import numpy as np
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint
from itertools import cycle, islice
import torch
import t3nsor as t3
from t3nsor import TensorTrainBatch
from dataloader import cifar_loader
from t3nsor import TensorTrain
import pdb


#construct sample covariance tensor and perform TT-SVD to get G_1,...,G_d
#return: list of G_1,G_2,...G_d TT-cores
def construct_sample_covariance_tensor(sample_mode, settings, mini_batch_input=None, mini_batch_target=None):
    #Use curr_minibatch to construct sample covariance tensor
    if sample_mode == 'curr_minibatch':
        #mini_batch_input shape for CIFAR-10: [batch_size 3 32 32]
        #pdb.set_trace()
        # mini_batch_target = torch.FloatTensor(mini_batch_target)
        # mini_batch_input = torch.FloatTensor(mini_batch_input)
        sample_cov = 1/settings.BATCH_SIZE * torch.einsum('bclw,b->clw', mini_batch_input.float(), mini_batch_target.float())
        #print('sample_cov shape:', sample_cov.shape) #should be [3,32,32]
        return t3.to_tt_tensor(sample_cov.view([3, 4, 8, 4, 8]), max_tt_rank=settings.TT_RANK).tt_cores

    # Use all training data to construct sample covariance tensor
    else:
        batch_size_cifar10 = 50000
        train_loader = cifar_loader(settings, 'train', batch_size=batch_size_cifar10, shuffle=True, data_augment=True)
        dataiter = iter(train_loader)
        raw_images, labels = dataiter.next()

        if settings.GPU:
            raw_images = raw_images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

        sample_cov = 1/batch_size_cifar10 * torch.einsum('bclw,b->clw', raw_images.float(), labels.float())
        #print('sample_cov shape:', sample_cov.shape)
        return t3.to_tt_tensor(sample_cov.view(3, 4, 8, 4, 8), max_tt_rank=settings.TT_RANK).tt_cores


#apply importance sketching to perform dimension reduction
#return list of dimension-reduced covariates, *have same dimension as TT cores*
def important_sketching(mini_batch, cov_tt_cores, settings):
    mini_batch = mini_batch.view(settings.BATCH_SIZE, 3, 4, 8, 4, 8)
    # assert(cov_tt_cores[0].shape[1] == 3)
    # assert(cov_tt_cores[1].shape[0] == 3)
    # assert(cov_tt_cores[2].shape[0] == 3)
    # assert(cov_tt_cores[3].shape[0] == 3)
    # assert(cov_tt_cores[4].shape[0] == 3)
    #print(cov_tt_cores[0])
    #print(cov_tt_cores[0].view(3, -1))
    fold_cov_tt_cores = [core.contiguous().view(3, -1) for core in cov_tt_cores]

    pinverse_cov_tt_cores = [torch.pinverse(core) for core in fold_cov_tt_cores]

    cov_list = []

    init_5_product = torch.einsum('bcdefg,gr->bcdefr', mini_batch, pinverse_cov_tt_cores[-1]).view(settings.BATCH_SIZE, 3, 4, 8, 4*settings.TT_RANK)
    init_4_product = torch.einsum('bcdef,fr->bcder', init_5_product, pinverse_cov_tt_cores[3]).view(settings.BATCH_SIZE, 3, 4, 8*settings.TT_RANK)
    init_3_product = torch.einsum('bcde,er->bcdr', init_4_product, pinverse_cov_tt_cores[2]).view(settings.BATCH_SIZE, 3, 4*settings.TT_RANK)
    x_1 = torch.einsum('bcd,dr->bcr', init_3_product, pinverse_cov_tt_cores[1])


    #no need to view, only for sake of test
    x_2 = torch.einsum('bcd,rc->brd', init_3_product, pinverse_cov_tt_cores[0]).contiguous().view(settings.BATCH_SIZE, settings.TT_RANK, 4, settings.TT_RANK)

    x_3 = torch.einsum('bcde,odj->bcoje', init_4_product, pinverse_cov_tt_cores[1].view(settings.TT_RANK, 4, settings.TT_RANK)).view(settings.BATCH_SIZE, 3, settings.TT_RANK, settings.TT_RANK, 8*settings.TT_RANK)
    #view for test
    x_3 = torch.einsum('bcrje,cr->bje', x_3, pinverse_cov_tt_cores[0]).view(settings.BATCH_SIZE, settings.TT_RANK, 8, settings.TT_RANK)

    x_4 = torch.einsum('bcdef, oej->bcdojf', init_5_product, pinverse_cov_tt_cores[2].view(settings.TT_RANK, 8, settings.TT_RANK)).view(settings.BATCH_SIZE, 3, 4, settings.TT_RANK, settings.TT_RANK, 4*settings.TT_RANK)
    x_4 = torch.einsum('bcdrof, jdr->bcjof', x_4, pinverse_cov_tt_cores[1].view(settings.TT_RANK, 4, settings.TT_RANK)).view(settings.BATCH_SIZE, 3, settings.TT_RANK, settings.TT_RANK, 4*settings.TT_RANK)
    x_4 = torch.einsum('bcrjf, cr->bjf', x_4, pinverse_cov_tt_cores[0]).view(settings.BATCH_SIZE, settings.TT_RANK, 4, settings.TT_RANK)

    x_5 = torch.einsum('bcdefg, ofj->bcdeojg', mini_batch, pinverse_cov_tt_cores[3].view(settings.TT_RANK, 4, settings.TT_RANK)).view(settings.BATCH_SIZE, 3, 4, 8, settings.TT_RANK, settings.TT_RANK,8)
    x_5 = torch.einsum('bcderog, jer->bcdjog', x_5, pinverse_cov_tt_cores[2].view(settings.TT_RANK, 8, settings.TT_RANK)).view(settings.BATCH_SIZE, 3, 4, settings.TT_RANK, settings.TT_RANK, 8)
    x_5 = torch.einsum('bcdrog, jdr->bcjog', x_5, pinverse_cov_tt_cores[1].view(settings.TT_RANK, 4, settings.TT_RANK)).view(settings.BATCH_SIZE, 3, settings.TT_RANK, settings.TT_RANK, 8)
    x_5 = torch.einsum('bcrog, cr->bog', x_5, pinverse_cov_tt_cores[0]).view(settings.BATCH_SIZE, settings.TT_RANK, 8)

    if settings.IS_OUTPUT_FORM == 'dense':
        #concatenate x_1,..,x_d to be the dimension reduced features
        cov_list.append(x_1.view(settings.BATCH_SIZE, -1))
        cov_list.append(x_2.view(settings.BATCH_SIZE, -1))
        cov_list.append(x_3.view(settings.BATCH_SIZE, -1))
        cov_list.append(x_4.view(settings.BATCH_SIZE, -1))
        cov_list.append(x_5.view(settings.BATCH_SIZE, -1))

        reduced_input = torch.cat(cov_list, 1)

    elif settings.IS_OUTPUT_FORM == 'tt_matrix':
        x_1 = torch.unsqueeze(x_1, dim=1)
        x_1 = torch.unsqueeze(x_1, dim=1)
        x_2 = torch.unsqueeze(x_2, dim=2)
        x_3 = torch.unsqueeze(x_3, dim=2)
        x_4 = torch.unsqueeze(x_4, dim=2)
        x_5 = torch.unsqueeze(x_5, dim=-1)
        x_5 = torch.unsqueeze(x_5, dim=2)

        cov_list.append(x_1)
        cov_list.append(x_2)
        cov_list.append(x_3)
        cov_list.append(x_4)
        cov_list.append(x_5)

        reduced_input = t3.TensorTrainBatch(cov_list)
        #pdb.set_trace()
    #assert(reduced_input.shape[0] == settings.BATCH_SIZE)
    #print('feature dimension:', reduced_input.shape[1])
    return reduced_input


def tt_to_dense(x):
    return x.full()


#GOAL: convert a [batch_size,1,512] tensor to TensorTrainBatch_matrix, 1 means output will be one TensorTrainBatch
def input_to_tt_matrix(input, settings):
    num_channels = input.shape[1]
    input_tt = []
    for num_c in range(num_channels):
        tt_cores_curr = []
        tt_batch_cores_curr = []

        for batch_iter in range(input.shape[0]):
            #settings.TT_MATRIX_SHAPE = [None,[4,8,4,4]]
            tt_cores_curr += t3.to_tt_matrix(input[batch_iter, num_c, :], shape=settings.TT_MATRIX_SHAPE, max_tt_rank=settings.TT_RANK).tt_cores

        tt_core_curr_unsq = [torch.unsqueeze(i, dim=0) for i in tt_cores_curr]
        for shift in range(1, 5):
            tt_batch_cores_curr.append(torch.cat(tt_core_curr_unsq[shift-1:(settings.BATCH_SIZE-1)*4+shift:4],dim=0))
        input_tt.append(TensorTrainBatch(tt_batch_cores_curr))
    return input_tt


def input_to_tt_tensor(inp, settings):
    #input shape: batch_size, num_channels, size_1, size_2
    #convert input from dense format to TT format, specifically, TensorTrainBatch
    #input shape: CIFAR-10: [batch_size 3 32 32] MINIST: [batch_size 1 28 28]
    assert(len(inp.shape) == 4)
    num_channels = inp.shape[1]
    input_tt = []
    for num_c in range(num_channels):
        tt_cores_curr = []
        tt_batch_cores_curr = []

        for batch_iter in range(settings.BATCH_SIZE):
            if settings.OTT:
                tt_cores_curr += t3.tt_to_ott(t3.to_tt_tensor(inp[batch_iter, num_c, :, :].view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK)).tt_cores
            else:
                #pdb.set_trace()
                tt_cores_curr += t3.to_tt_tensor(inp[batch_iter, num_c, :, :].view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK).tt_cores

        tt_core_curr_unsq = [torch.unsqueeze(i, dim=0) for i in tt_cores_curr]
        assert(len(settings.TT_SHAPE) == 4)
        for shift in range(1, 5):
            tt_batch_cores_curr.append(torch.cat(tuple(tt_core_curr_unsq[shift-1:(settings.BATCH_SIZE-1)*4+shift:4]), dim=0))
        input_tt.append(TensorTrainBatch(tt_batch_cores_curr))
    #pdb.set_trace()
    return input_tt


#use after input_to_tt_tensor, output: tt_rgb 'image'
# def tt_rgb_image(ttbatch_list, settings):
#     return

def b_inv33(b_mat):
    #b_mat = b_mat.cpu()
    #eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    #b_inv, _ = torch.gesv(eye, b_mat)
    #b_inv = b_inv.to(device)
    #print(b_inv.contiguous())
    #b = [t.inverse() for t in torch.unbind(b_mat)]
    #b_inv = torch.stack(b)
    b00 = b_mat[:,0,0]
    b01 = b_mat[:,0,1]
    b02 = b_mat[:,0,2]
    b10 = b_mat[:,1,0]
    b11 = b_mat[:,1,1]
    b12 = b_mat[:,1,2]
    b20 = b_mat[:,2,0]
    b21 = b_mat[:,2,1]
    b22 = b_mat[:,2,2]
    det = (b00*(b11*b22-b12*b21)-b01*(b10*b22-b12*b20)+b02*(b10*b21-b11*b20))
    c00 = b11*b22 - b12*b21
    c01 = b02*b21 - b01*b22
    c02 = b01*b12 - b02*b11
    c10 = b12*b20 - b10*b22
    c11 = b00*b22 - b02*b20
    c12 = b02*b10 - b00*b12
    c20 = b10*b21 - b11*b20
    c21 = b01*b20 - b00*b21
    c22 = b00*b11 - b01*b10
    eps = 1e-5
    c00 = (c00/ (det+eps)).view(-1, 1, 1)
    c01 = (c01/ (det+eps)).view(-1, 1, 1)
    c02 = (c02/ (det+eps)).view(-1, 1, 1)
    c10 = (c10/ (det+eps)).view(-1, 1, 1)
    c11 = (c11/ (det+eps)).view(-1, 1, 1)
    c12 = (c12/ (det+eps)).view(-1, 1, 1)
    c20 = (c20/ (det+eps)).view(-1, 1, 1)
    c21 = (c21/ (det+eps)).view(-1, 1, 1)
    c22 = (c22/ (det+eps)).view(-1, 1, 1)
    b_inv1 = torch.cat((torch.cat((c00,c01,c02), dim=2), torch.cat((c10,c11,c12), dim=2), torch.cat((c20,c21,c22), dim=2)), dim=1)
    return b_inv1


def b_inv(b_mat):
    #b_mat = b_mat.cpu()
    #eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    #b_inv, _ = torch.gesv(eye, b_mat)
    #b_inv = b_inv.to(device)
    #print(b_inv.contiguous())
    #b = [t.inverse() for t in torch.unbind(b_mat)]
    #b_inv = torch.stack(b)
    eps = 1e-5
    b00 = b_mat[:,0,0]
    b01 = b_mat[:,0,1]
    b10 = b_mat[:,1,0]
    b11 = b_mat[:,1,1]
    #elementwise multiplication and division
    det = (b00*b11-b01*b10)
    b00 = b00/ (det+eps)
    b01 = b01/ (det+eps)
    b10 = b10/ (det+eps)
    b11 = b11/ (det+eps)
    b_inv1 = torch.cat((torch.cat((b11.view(-1,1,1),-1.*b01.view(-1,1,1)),dim=2),torch.cat((-1.*b10.view(-1,1,1),b00.view(-1,1,1)),dim=2)),dim=1)
    return b_inv1


#shape of t: objk [out_channels,batch_size,r_1,r_2]
#Q = (I-A)*inv(I+A)
#inverse cayley: A = (I-Q)*inv(I+Q)
def cayley(t,gpu=True):
    if gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if len(t.shape) != 4:
        print('wrong dimension!')
    for o in range(0,t.shape[0]):
        #for b in range(0,t.shape[1]):
        #    t[o,b,:,:] = torch.matmul( torch.eye(t.shape[2],t.shape[3]).to(device) - t[o,b,:,:] , torch.inverse(torch.eye(t.shape[2],t.shape[3]).to(device) + t[o,b,:,:]) )
        batch_identity = torch.unsqueeze(torch.eye(t.shape[2],t.shape[3]),0).repeat(t.shape[1],1,1).to(device)
        #The matrix multiplication is always done with using the last two dimensions. All the ones before are considered as batch.
        t[o,:,:,:] = torch.matmul(batch_identity-t[o,:,:,:],torch.inverse(batch_identity+t[o,:,:,:]))
    return t


MODES = ['ascending', 'descending', 'mixed']
CRITERIONS = ['entropy', 'var']

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def _to_list(p):
    res = []
    for k, v in p.items():
        res += [k, ] * v
    return res


def _roundup(n, k):
    return int(np.ceil(n / 10**k)) * 10**k


def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def _get_all_factors(n, d=3, mode='ascending'):
    p = _factorint2(n)
    if len(p) < d:
        p = p + [1, ] * (d - len(p))

    if mode == 'ascending':
        def prepr(x):
            return tuple(sorted([np.prod(_) for _ in x]))
    elif mode == 'descending':
        def prepr(x):
            return tuple(sorted([np.prod(_) for _ in x], reverse=True))

    elif mode == 'mixed':
        def prepr(x):
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[:N//2], x[N//2:]
            return tuple(_roundrobin(xf, xl))

    else:
        raise ValueError('Wrong mode specified, only {} are available'.format(MODES))

    raw_factors = multiset_partitions(p, d)
    clean_factors = [prepr(f) for f in raw_factors]
    clean_factors = list(set(clean_factors))
    return clean_factors


def _factorint2(p):
    return _to_list(factorint(p))


def auto_shape(n, d=3, criterion='entropy', mode='ascending'):
    factors = _get_all_factors(n, d=d, mode=mode)
    if criterion == 'entropy':
        weights = [entropy(f) for f in factors]
    elif criterion == 'var':
        weights = [-np.var(f) for f in factors]
    else:
        raise ValueError('Wrong criterion specified, only {} are available'.format(CRITERIONS))

    i = np.argmax(weights)
    return list(factors[i])


def suggest_shape(n, d=3, criterion='entropy', mode='ascending'):
    weights = []
    for i in range(len(str(n))):

        n_i = _roundup(n, i)
        if criterion == 'entropy':
            weights.append(entropy(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
        elif criterion == 'var':
            weights.append(-np.var(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
        else:
            raise ValueError('Wrong criterion specified, only {} are available'.format(CRITERIONS))

    i = np.argmax(weights)
    factors = auto_shape(int(_roundup(n, i)), d=d, mode=mode, criterion=criterion)
    return factors


def svd_fix(x):
    n = x.shape[0]
    m = x.shape[1]

    if n > m:
        u, s, v = torch.svd(x)

    else:
        u, s, v = torch.svd(x.t())
        v, u = u, v
    #pdb.set_trace()
    return u, s, v


def ind2sub(siz, idx):
    n = len(siz)
    b = len(idx)
    subs = []
    k = np.cumprod(siz[:-1])
    k = np.concatenate((np.ones(1), k))

    for i in range(n - 1, -1, -1):
        subs.append(torch.floor(idx.float() / k[i]).long())
        idx = torch.fmod(idx, k[i])

    return torch.stack(subs[::-1], dim=1)


def is_batch_broadcasting_possible(tt_a, tt_b):
  """Check that the batch broadcasting possible for the given batch sizes.
  Returns true if the batch sizes are the same or if one of them is 1.
  If the batch size that is supposed to be 1 is not known on compilation stage,
  broadcasting is not allowed.
  Args:
    tt_a: TensorTrain or TensorTrainBatch
    tt_b: TensorTrain or TensorTrainBatch
  Returns:
    Bool
  """
  try:
    if tt_a.batch_size is None and tt_b.batch_size is None:
      # If both batch sizes are not available on the compilation stage,
      # we cannot say if broadcasting is possible so we will not allow it.
      return False
    if tt_a.batch_size == tt_b.batch_size:
      return True
    if tt_a.batch_size == 1 or tt_b.batch_size == 1:
      return True
    return False
  except AttributeError:
    # One or both of the arguments are not batch tensor, but single TT tensors.
    # In this case broadcasting is always possible.
    return True

#opposite to expand_batch_dim
def squeeze_batch_dim(tt, name='t3f_squeeze_batch_dim'):
  """Converts batch size 1 TensorTrainBatch into TensorTrain.
  Args:
    tt: TensorTrain or TensorTrainBatch.
    name: string, name of the Op.
  Returns:
    TensorTrain if the input is a TensorTrainBatch with batch_size == 1 (known
      at compilation stage) or a TensorTrain.
    TensorTrainBatch otherwise.
    """
    try:
      if tt.batch_size == 1:
          #Liancheng
          tt_cores = []
          for core_idx in range(tt.ndims):
              curr_core = torch.squeeze(tt.tt_cores[core_idx], dim=0)
              tt_cores.append(curr_core)
        return t3.TensorTrain(tt_cores)
      else:
        return tt
    except AttributeError:
      # tt object does not have attribute batch_size, probably already
      # a TensorTrain.
      return tt


def tt_batch_from_list_of_tt(tt_list):
    expanded_tt_list = [expand_batch_dim(tt) for tt in tt_list]
    tt_batch_cores = []
    for core_idx in range(tt_list[0].ndims):
        for tt_idx in range(len(tt_list)):
            if tt_idx = 0
                cumu_tt_core = expand_tt_list[tt_idx].tt_cores[core_idx]
            else:
                cumu_tt_core = torch.cat((cumu_tt_core, expand_tt_list[tt_idx].tt_cores[core_idx]), 0)
        tt_batch_cores.append(cumu_tt_core)
    return TensorTrainBatch(tt_batch_cores)


def expand_batch_dim(tt, name='t3f_expand_batch_dim'):
  """Creates a 1-element TensorTrainBatch from a TensorTrain.
  Args:
    tt: TensorTrain or TensorTrainBatch.
    name: string, name of the Op.
  Returns:
    TensorTrainBatch
  """

if hasattr(tt, 'batch_size'):
    return tt
else:
    tt_cores = []
    for core_idx in range(tt.ndims):
        tt_cores.append(torch.unsqueeze(tt.tt_cores[core_idx], dim=0))
    return TensorTrainBatch(tt_cores)


def get_element_from_batch(tt_batch, idx):
    """get d th tt element from tt_batch"""
    tt_cores = []
    if not hasattr(tt_batch, 'batch_size'):
        print('input not a tt batch!')
    if tt_batch.is_tt_matrix:
        for core_idx in range(tt_batch.ndims):
            tt_cores.append(tt_batch.tt_cores[core_idx][idx,:,:,:,:])
    else:
        for core_idx in range(tt_batch.ndims):
            tt_cores.append(tt_batch.tt_cores[core_idx][idx,:,:,:])
    return TensorTrain(tt_cores)
