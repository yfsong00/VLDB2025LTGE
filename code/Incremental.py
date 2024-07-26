from scipy.linalg import qr
from scipy.linalg import svd
import scipy
import math
import numpy as np
import time

def Increse(embedding, V, edges, node_index, num_per_motif, dimension = 32):
    #pca = PCA(n_components = dimension)
    #print(num_per_motif)
    #print(len(edges))
    T = edges[-1][2]-edges[0][2]
    E = np.zeros([int(node_index)+1,int(T/num_per_motif+1)])
    #E = scipy.sparse.lil_matrix((int(node_index)+1,int(T/num_per_motif+1)))
    label = 0
    tag = 0
    for l in range(len(edges)):
        if edges[l][2]-edges[tag][2]-num_per_motif > 0.0:
            tag = l
            label+=1
        # E[int(edges[l][0]),label]+=max(1,edges[l][2])
        # E[int(edges[l][1]),label]+=max(1,edges[l][2])
        E[int(edges[l][0])][label]+=max(1,edges[l][2])
        E[int(edges[l][1])][label]+=max(1,edges[l][2])
    #E = normalize(E, norm = 'l2', axis=0)
    # for i, row in enumerate(E.rows):
    #     for j in row:
    #         if E[i, j] > 0:  # 确保元素大于零
    #             E[i, j] = np.log(E[i, j])
    E = np.log(E, out=np.zeros_like(E), where=(np.isclose(E, 0.0)==False))
    #k=min(dimension,E.shape[1]-1)
    #E = E.toarray()
    #U2, s2, V2 = scipy.sparse.linalg.svds(E, k=k)
    U2, s2, V2 = np.linalg.svd(E, full_matrices=0)
    Sigma2 = np.diag(s2) 
    UH2 = U2@Sigma2#E
    VH2 = V2#np.eye(E.shape[1])
    if len(embedding) < len(UH2):
        Zeros_e = np.zeros((len(UH2)-len(embedding),len(embedding[0])))
        embedding = np.vstack((embedding,Zeros_e))
    elif len(embedding) > len(UH2):
        Zeros_e = np.zeros((len(embedding)-len(UH2),len(UH2[0])))
        UH2 = np.vstack((UH2,Zeros_e))
    #SigmaH2 = np.diag(SigmaH2)
    Q1, R1 = np.linalg.qr(np.hstack((embedding, UH2)))
    length1, height1 = V.T.shape
    length2, height2 = VH2.T.shape
    Zeros1 = np.zeros((length1, height2))
    Zeros2 = np.zeros((length2, height1))
    V = np.hstack((V.T,Zeros1))
    VH2 = np.hstack((Zeros2,VH2.T))
    Q2, R2 = np.linalg.qr(np.vstack((V, VH2)))
    #print(R1.shape)
    #print(R2.shape)
    if R2.shape[1] > R1.shape[1]:
        Zeros3 = np.zeros((R1.shape[0], R2.shape[1]-R1.shape[1]))
        R1 = np.hstack((R1,Zeros3))

    S = R1@R2.T
    Ur, Sigmar, Vr = np.linalg.svd(S, full_matrices = 0)
    SigmarH = np.diag(Sigmar)
    S = Ur@SigmarH
    return (Q1@S)[:,:dimension], (Vr@Q2.T)[:dimension,:]