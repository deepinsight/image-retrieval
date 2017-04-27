import numpy as np

class SecondOrder:

  @staticmethod
  def raid_g(fdata, norm = True):
    features = None
    if len(fdata.shape)==4: #which from CNN layer directly
      fdata = np.transpose(fdata, (0,2,3,1))
      features = fdata[0,:,:,:].reshape( (fdata.shape[1]*fdata.shape[2],fdata.shape[3]) )
    else:
      features = fdata
    features = np.sign(features)*(np.abs(features)**0.5)
    mean = np.mean(features, axis=0)
    cov = np.cov(np.transpose(features))
    beta = 0.3

    alpha = 0.75
    gama = (1-alpha)/(2*alpha);
    for i in xrange(cov.shape[0]):
      cov[i][i] += 0.001
    U, s, V = np.linalg.svd(cov)
    diag_s = np.diag(s);
    _x1 = np.abs(diag_s/alpha)+gama*gama
    _x2 = _x1**0.55 - gama

    diag_s = np.diag(np.diag(np.sign(diag_s)*_x2))
    cov = np.dot(U, diag_s).dot(U.T)
    #print(cov)


    _u = mean.reshape( (len(mean),1) )
    _ut = np.transpose(_u)
    uu = np.dot(_u, _ut)
    s_matrix = np.hstack( (np.vstack((cov+beta*beta*uu, beta*_ut)),
        np.vstack( (beta*_u, np.ones((1,1), dtype=np.float)) ) ) 
        )
    k = s_matrix.shape[0]
    iu = np.triu_indices(k)
    #_aaa = np.isnan(s_matrix)
    #_bbb = np.isinf(s_matrix)
    #print(np.any(_aaa))
    #print(np.any(_bbb))


    [U, s, V] =np.linalg.svd(s_matrix);
    #print(U.shape)
    diag_s = np.diag(s);
    diag_s = diag_s + 1e-16
    _x1 = np.abs(diag_s)**0.9
    log_diag_s = np.diag(np.diag(np.sign(diag_s)*_x1))
    g_matrix = U.dot(log_diag_s).dot(U.T)


    rfeature = g_matrix[iu]
    if norm:
      rfeature /= np.linalg.norm(rfeature)
    return rfeature

