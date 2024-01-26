from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
import torch
def one_hot(x, K):
    """Turn labels x into one hot vector of K classes. """
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    # Pi = np.zeros(shape=(num_classes, 512, 512))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

class Metric():
    def __init__(self, mode, num_classes=None,**kwargs):
        self.gam1 = 1
        self.gam2 = 1
        self.eps = 0.5
        self.num_classes = num_classes
        self.requires = ['embeds','target_labels']
        self.mode = mode
        self.name = 'coding_rate@{}'.format(mode)

    def __call__(self, embeds, target_labels):
        embeds = embeds[:5000, :]
        target_labels = target_labels[:5000, :]
        target_labels = target_labels.flatten()

        W = embeds.T
        with torch.no_grad():
            if 'dis' in self.mode:
                p, m = W.shape
                I = torch.eye(p).cuda()
                scalar = p / (m * self.eps)
                logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
                discrimn_coding_rate = (logdet / 2).cpu()
                coding_rate = discrimn_coding_rate
            if 'com' in self.mode:
                # Pi = label_to_membership(target_labels.numpy(), self.num_classes)
                Pi = label_to_membership(target_labels, self.num_classes)
                Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
                p, m = W.shape
                k, k1, k2 = Pi.shape
                I = torch.eye(p).cuda()
                compress_cr = 0.
                for j in range(k):
                    trPi = torch.trace(Pi[j]) + 1e-8
                    scalar = p / (trPi * self.eps)
                    log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
                    compress_cr += log_det * trPi / m
                compress_coding_rate = (compress_cr / 2).cpu()
                coding_rate = compress_coding_rate
            if 'delta_dis_and_com' == self.mode:
                # delta_codingrate = self.gam2 * -discrimn_coding_rate + compress_coding_rate
                delta_codingrate = self.gam2 * discrimn_coding_rate - compress_coding_rate
                coding_rate = delta_codingrate
        # del W, I, Pi, logdet, discrimn_coding_rate, compress_cr, compress_coding_rate, delta_codingrate
        torch.cuda.empty_cache()
        return coding_rate






