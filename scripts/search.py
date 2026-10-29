import numpy as np


def search_vectors(txt_vec, img_vecs, k=5):
    scores = (txt_vec @ img_vecs.T)[0]
    inds = np.argsort(-scores, kind='stable')[:k]
    
    return scores[inds], inds