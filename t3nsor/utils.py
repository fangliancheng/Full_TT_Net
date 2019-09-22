from scipy.stats import entropy
import numpy as np
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint
from itertools import cycle, islice
import torch

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
        t[o,:,:,:] = torch.matmul(batch_identity-t[o,:,:,:],b_inv(batch_identity+t[o,:,:,:]))
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
