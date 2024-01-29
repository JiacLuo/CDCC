import numpy as np
import torch
def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def remove_frequency(x, maskout_ratio=0):
    mask = torch.FloatTensor(x.shape).uniform_() > maskout_ratio
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0,):
    mask = torch.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape).to(x.device)*(max_amplitude*0.1).to(x.device)#不知道为什么报错
    pertub_matrix = mask*random_am
    return x+pertub_matrix

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def DataTransform_T(data,model_params):
    aug_1 = jitter(data, model_params.jitter_ratio)
    aug_2 = scaling(data, model_params.jitter_scale_ratio)
    aug_3 = permutation(data, max_segments=model_params.max_seg)
    aug_2=torch.from_numpy(aug_2)
    li = np.random.randint(0, 3, size=[data.shape[0]])
    li_onehot = one_hot_encoding(li)
    # Sampling from three data augmentation methods
    aug_1[1-li_onehot[:, 0]] = 0
    aug_2[1 - li_onehot[:, 1]] = 0
    aug_3[1 - li_onehot[:, 2]] = 0
    aug_T = aug_1 + aug_2 + aug_3
    return data,aug_T

def DataTransform_F(sample, model_params):
    # https://arxiv.org/pdf/2206.08496.pdf
    aug_1 =  remove_frequency(sample, model_params.remove_frequency_ratio)
    aug_2 = add_frequency(sample, model_params.add_frequency_ratio)
    li = np.random.randint(0, 2, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    # Sampling from two data augmentation methods
    aug_1[1-li_onehot[:, 0]] = 0
    aug_2[1 - li_onehot[:, 1]] = 0
    aug_F = aug_1 + aug_2
    return sample,aug_F

