
import torch
import torch.nn as nn
import numpy as np
import sys
from datetime import datetime, timedelta

beta_sampler = torch.distributions.Beta(2., 2.)

class SinkBF_gauss(nn.Module):
    def __init__(self, Nx, Nw, Ny):
        super(SinkBF_gauss, self).__init__()
        self.Nx = Nx
        self.Nw = Nw
        self.Ny = Ny

        self.q_x_y = nn.Sequential(nn.Linear(Ny, Nx), nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.q_w_xy = nn.Sequential(nn.Linear(Nx+Ny, Nw), nn.Tanh(),
            nn.Linear(Nw, Nw),)
        self.f_x_xw = nn.GRU(Nw, Nx, 1) # num_layers = 1

    def forward(self, _Y0, Nhrzn):
# _Y0: (Nwup+1, *, Ny)
        Nx, Nw = self.Nx, self.Nw

        assert _Y0.shape[0] >= 1, "Y0 is not allowed to be empty."
        Nwup = _Y0.shape[0] - 1
        Nbatch = _Y0.shape[1]

        _x = self.q_x_y(_Y0[0,:]) # (*, Nx)
        X = [_x,]
        W = []
        for t in range(1, Nwup+1+Nhrzn):
            if t <= Nwup:
                _w = self.q_w_xy(torch.cat((_x, _Y0[t,:]), dim=1)) # (*, Nw)
                W.append(_w)
            else:
                if self.training:
                    _w = torch.zeros(Nbatch, Nw)
                else:
                    _w = torch.randn(Nbatch, Nw)
            _x = self.f_x_xw(_w.unsqueeze(0), _x.unsqueeze(0))[1][0,:] # (*, Nx)
            X.append(_x)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        if len(W) > 0:
            _W = torch.stack(W, dim = 0) # (Nwup, *, Nw)
        else:
            _W = torch.zeros(Nwup, Nbatch, Nw)
        _Yhat = self.p_y_x(_X) # (Nwup+1+Nhrzn, *, Ny)

        return _X, _W, _Yhat
        


def robust_sinkhorn_iteration(_M, _p, _q, _eps, tol, max_itr):
    _alpha = _p * 0
    _beta = _q * 0
    cnt = 0
    
    assert max(_M.shape) <= 2**7, "The shape of M is %s. That exceeds the limitation: 64" % str(_M.shape)

    while True:
        
        _P = torch.exp(-(_M-_alpha-_beta)/_eps -1)
        _qhat = torch.sum(_P, dim=0, keepdim=True)
        _err = torch.sum(torch.abs(_qhat - _q))

        if _err < tol or cnt >= max_itr:
            break
        else:
            cnt += 1

        _delta_row = torch.min(_M - _alpha, dim=0, keepdim = True)[0]
        _beta = _eps + _eps * torch.log(_q) + _delta_row  \
            -_eps * torch.log( torch.sum( \
            torch.exp(-(_M-_alpha-_delta_row)/_eps),\
            dim=0, keepdim = True ) )

        _delta_col = torch.min(_M - _beta, dim=1, keepdim = True)[0]
        _alpha = _eps + _eps * torch.log(_p) + _delta_col \
            -_eps * torch.log( torch.sum( \
            torch.exp( -(_M - _beta - _delta_col)/_eps),\
            dim=1, keepdim = True )  )

    #_dist = torch.sum(_p * _alpha) + torch.sum(_q * _beta) - _eps
    _dist = torch.sum(_P * _M)
    return _dist, cnt

def measure_distance(_X0, _X1, tol = 1e-4, eps_given = 1e-2, max_itr = 2**5, 
    norm_p = 1):
    # _X0: (*, Nx), _X1: (*, Nx)

    assert norm_p >=1 and isinstance(norm_p, int)
    if norm_p == 1:
        _M01 = torch.mean(torch.abs(_X0.unsqueeze(1) - _X1), dim=2)
    else:
        _M01 = torch.mean((_X0.unsqueeze(1) - _X1)**norm_p, dim=2)

    _p = 1/_X0.shape[0] * torch.ones(_X0.shape[0])
    _q = 1/_X1.shape[0] * torch.ones(_X1.shape[0])

    _eps = torch.tensor(eps_given) * float(torch.mean(_M01))
        
    _dist01, cnt01 = robust_sinkhorn_iteration(_M01, _p.unsqueeze(1), 
        _q.unsqueeze(0), _eps, tol, max_itr=max_itr)

    return _dist01, cnt01


def run_training_gauss(sbf, data_generator, optimizer, Nepoch, Nbatch, Nwup, Nhrzn,
    reg_param, eps_given = 1e-2):
    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch
    Nx, Nw, Ny = sbf.Nx, sbf.Nw, sbf.Ny

    t_bgn = datetime.now()
    for epoch in range(Nepoch):
        loss_hist = []
        for k1 in range(Nitr):
            sys.stdout.write('%03d/%03d[EOL]\r' % (k1, Nitr))
            Ybatch = data_generator.batch(Nbatch, Nwup+1+Nhrzn)
            _Y0batch = torch.tensor(Ybatch[:Nwup+1,:]) # (Nwup+1, *, Ny)
            _Ybatch = torch.tensor(Ybatch) # (Nwup+1+Nhrzn, *, Ny)

            _Xbatch, _Wbatch, _Yhat_batch = sbf(_Y0batch, Nhrzn)
            _resid = torch.mean((_Yhat_batch - _Ybatch)**2)

            _X0 = _Xbatch[0,:] # (*, Nx)
            _reg_term, _ = measure_distance(_X0, torch.randn(Nbatch, Nx), 
                eps_given = eps_given)

            for k1 in range(Nwup):
                _reg_term += measure_distance(_Wbatch[k1,:], 
                    torch.randn(Nbatch, Nw),
                    eps_given = eps_given)[0]
            _reg_term /= (Nwup+1)
            
            _loss = _resid + reg_param * _reg_term
            loss_hist.append((float(_resid), float(_reg_term), float(_loss)))

            sbf.zero_grad()
            _loss.backward()
            optimizer.step()

        resid_avg, reg_term_avg, loss_avg = np.mean(loss_hist, axis=0)
        sys.stdout.write('%s epoch %04d resid %8.2e reg term %8.2e loss %8.2e\n' %
            (datetime.now() - t_bgn, epoch+1, resid_avg, reg_term_avg, loss_avg))
    

class SinkBF_beta(nn.Module):
    def __init__(self, Nx, Nw, Ny):
        super(SinkBF_beta, self).__init__()
        self.Nx = Nx
        self.Nw = Nw
        self.Ny = Ny

        self.logit_q_x_y = nn.Sequential(nn.Linear(Ny, Nx), nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.logit_q_w_xy = nn.Sequential(nn.Linear(Nx+Ny, Nw), nn.Tanh(),
            nn.Linear(Nw, Nw),)
        #self.f_x_xw = nn.GRU(Nw, Nx, 1) # num_layers = 1
        self.f_x_xw = nn.Sequential(nn.Linear(Nw+Nx, Nx), nn.Tanh(),) # num_layers = 1

        self._log_alpha = nn.Parameter(torch.rand(Nx))

    def forward(self, _Y0, Nhrzn):
# _Y0: (Nwup+1, *, Ny)
        Nx, Nw = self.Nx, self.Nw

        assert _Y0.shape[0] >= 1, "Y0 is not allowed to be empty."
        Nwup = _Y0.shape[0] - 1
        Nbatch = _Y0.shape[1]

        _alpha = torch.exp(-torch.abs(self._log_alpha))
        _beta = torch.sqrt(1-_alpha**2)

        _x = torch.tanh(self.logit_q_x_y(_Y0[0,:])/2) # (*, Nx)
        X = [_x,]
        W = []
        for t in range(1, Nwup+1+Nhrzn):
            if t <= Nwup:
                _logit_q_w = self.logit_q_w_xy(torch.cat((_x, _Y0[t,:]), \
                    dim=1)) # (*, Nw)
                _w = torch.tanh(_logit_q_w/2) # (*, Nw)
                W.append(_w)
            else:
                if self.training:
                    _w = torch.zeros(Nbatch, Nw) 
                else:
                    _w = 2 * beta_sampler.sample((Nbatch, Nw)) - 1
            #_x = self.f_x_xw(_w.unsqueeze(0), _x.unsqueeze(0))[1][0,:] # (*, Nx)
            _x = _alpha * _x \
                + _beta * self.f_x_xw(torch.cat((_x, _w), dim=1)) # (*, Nx) 
            X.append(_x)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        _W = torch.stack(W, dim = 0) # (Nwup, *, Nw)
        _Yhat = self.p_y_x(_X) # (Nwup+1+Nhrzn, *, Ny)

        return _X, _W, _Yhat


def run_training_beta(sbf_beta, data_generator, optimizer, Nepoch, Nbatch, \
    Nwup, Nhrzn, reg_param, eps_given = 1e-2):
    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch
    Nx, Nw, Ny = sbf_beta.Nx, sbf_beta.Nw, sbf_beta.Ny

    t_bgn = datetime.now()
    for epoch in range(Nepoch):
        loss_hist = []
        for k1 in range(Nitr):
            Ybatch = data_generator.batch(Nbatch, Nwup+1+Nhrzn)
            _Y0batch = torch.tensor(Ybatch[:Nwup+1,:]) # (Nwup+1, *, Ny)
            _Ybatch = torch.tensor(Ybatch) # (Nwup+1+Nhrzn, *, Ny)

            _Xbatch, _W, _Yhat_batch = sbf_beta(_Y0batch, Nhrzn)
            _resid = torch.mean((_Yhat_batch - _Ybatch)**2)

            _X0 = _Xbatch[0,:] # (*, Nx)
            _discrepancy_q, _ = measure_distance(_X0, 
                2*beta_sampler.sample((Nbatch, Nx))-1, eps_given = eps_given)

            for k1 in range(Nwup):
                _discrepancy_q += measure_distance(_W[k1,:], 
                    2*beta_sampler.sample((Nbatch, Nw))-1, 
                    eps_given = eps_given)[0]
            _discrepancy_q /= (Nwup+1)

            _loss = _resid + reg_param * _discrepancy_q
            loss_hist.append((float(_resid), float(_discrepancy_q), 
                float(_loss)))

            sbf_beta.zero_grad()
            _loss.backward()
            optimizer.step()

        resid_avg, reg_term_avg, loss_avg = np.mean(loss_hist, axis=0)
        sys.stdout.write('%s epoch %04d resid %8.2e reg term %8.2e loss %8.2e\r' %
            (datetime.now() - t_bgn, epoch+1, resid_avg, reg_term_avg, loss_avg))


def run_training_002(sbf_guass, data_generator, optimizer, Nepoch, Nbatch, \
    Nwup, Nhrzn, reg_param, eps_given = 1e-1, tol = 1e-2, max_itr = 2**10, 
    loss_hist = None, rand = 'gauss'):

    if loss_hist is None:
        loss_hist = []

    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch
    Nx, Nw, Ny = sbf_guass.Nx, sbf_guass.Nw, sbf_guass.Ny

    t_bgn = datetime.now()
    for epoch in range(Nepoch):
        tmp = []
        for k1 in range(Nitr):
            Ybatch = data_generator.batch(Nbatch, Nwup+1+Nhrzn)
            _Y0batch = torch.tensor(Ybatch[:Nwup+1,:]) # (Nwup+1, *, Ny)
            _Ybatch = torch.tensor(Ybatch) # (Nwup+1+Nhrzn, *, Ny)

            _Xbatch, _W, _Yhat_batch = sbf_guass(_Y0batch, Nhrzn)
# _Xbatch: (Nwup+1+Nhrzn, *, Nx), _W: (Nwup, *, Nw), 
# _Yhat_batch: (Nwup+1+Nhrzn, *, Ny)
            _resid = torch.mean((_Yhat_batch - _Ybatch)**2)

            if Nwup > 0:

                idx0 = torch.randint(0, Nwup, size= (()))
                idx1 = torch.randint(0, Nbatch, size = (()))
                idx2 = torch.randint(0, Nw, size = (()))

                _testW0 = _W[:, idx1, idx2].reshape(-1,1) # (Nwup, 1)
                _testW1 = _W[idx0, :, idx2].reshape(-1,1) # (Nwup, 1)

                if rand == "gauss":
                    _refW0 = torch.randn(2**7, 1) # (2**7, 1)
                    _refW1 = torch.randn(2**7, 1) # (2**7, 1)

                if rand == "uniform":
                    _refW0 = 2*torch.rand(2**7, 1)-1 # (2**7, 1)
                    _refW1 = 2*torch.rand(2**7, 1)-1 # (2**7, 1)

                _dist0, cnt = measure_distance(_testW0, _refW0, tol = tol, 
                    eps_given = eps_given, max_itr = max_itr, norm_p = 2)

                _dist1, cnt = measure_distance(_testW1, _refW1, tol = tol, 
                    eps_given = eps_given, max_itr = max_itr, norm_p = 2)

                _dist = (_dist0 + _dist1)/2
            else:
                _dist = torch.zeros(())

            _loss = _resid + reg_param * _dist
            tmp.append((float(_loss), float(_resid), float(_dist)))

            sbf_guass.zero_grad()
            _loss.backward()
            optimizer.step()

        loss_avg, resid_avg, dist_avg = np.mean(tmp, axis=0)
        elapsed_time = datetime.now() - t_bgn
        sys.stdout.write('%s epoch %04d loss %8.2e resid %8.2e dist %8.2e\r' %
            (elapsed_time, epoch, loss_avg, resid_avg, dist_avg))

        loss_hist.append((elapsed_time, epoch, loss_avg, resid_avg, dist_avg,))

    return loss_hist


def run_training_003(sbf_guass, data_generator, optimizer, Nepoch, Nbatch, \
    Nwup, Nhrzn, reg_param, eps_given = 1e-1, tol = 1e-2, max_itr = 2**10, 
    loss_hist = None):

    if loss_hist is None:
        loss_hist = []

    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch
    Nx, Nw, Ny = sbf_guass.Nx, sbf_guass.Nw, sbf_guass.Ny

    t_bgn = datetime.now()
    for epoch in range(Nepoch):
        tmp = []
        for k1 in range(Nitr):
            Ybatch = data_generator.batch(Nbatch, Nwup+1+Nhrzn)
            _Y0batch = torch.tensor(Ybatch[:Nwup+1,:]) # (Nwup+1, *, Ny)
            _Ybatch = torch.tensor(Ybatch) # (Nwup+1+Nhrzn, *, Ny)

            _Xbatch, _W, _Yhat_batch = sbf_guass(_Y0batch, Nhrzn)
# _Xbatch: (Nwup+1+Nhrzn, *, Nx), _W: (Nwup, *, Nw), 
# _Yhat_batch: (Nwup+1+Nhrzn, *, Ny)
            _resid = torch.mean((_Yhat_batch - _Ybatch)**2)

            idx0 = torch.randint(0, Nwup, size= (()))
            idx1 = torch.randint(0, Nbatch, size = (()))
            idx2 = torch.randint(0, Nw, size = (()))

            _testW0 = _W[:, idx1, idx2].reshape(-1,1) # (Nwup, 1)
            _refW0 = torch.randn(2**7, 1) # (2**7, 1)

            _testW1 = _W[idx0, :, idx2].reshape(-1,1) # (Nwup, 1)
            _refW1 = torch.randn(2**7, 1) # (2**7, 1)

            _testW2 = torch.sum(_W[:idx0, :, idx2], dim=0).reshape(-1,1)\
                /np.sqrt(idx0+1) # (*, 1)
            _refW2 = torch.randn(2**7, 1) # (2**7, 1)

            _dist0, cnt = measure_distance(_testW0, _refW0, tol = tol, 
                eps_given = eps_given, max_itr = max_itr, norm_p = 2)

            _dist1, cnt = measure_distance(_testW1, _refW1, tol = tol, 
                eps_given = eps_given, max_itr = max_itr, norm_p = 2)

            _dist2, cnt = measure_distance(_testW2, _refW2, tol = tol, 
                eps_given = eps_given, max_itr = max_itr, norm_p = 2)

            _dist = (_dist0 + _dist1 + _dist2)/3

            _loss = _resid + reg_param * _dist
            tmp.append((float(_loss), float(_resid), float(_dist)))

            sbf_guass.zero_grad()
            _loss.backward()
            optimizer.step()

        loss_avg, resid_avg, dist_avg = np.mean(tmp, axis=0)
        elapsed_time = datetime.now() - t_bgn
        sys.stdout.write('%s epoch %04d loss %8.2e resid %8.2e dist %8.2e\r' %
            (elapsed_time, epoch, loss_avg, resid_avg, dist_avg))

        loss_hist.append((elapsed_time, epoch, loss_avg, resid_avg, dist_avg,))

    return loss_hist


class SinkBF_moving_average(nn.Module):
    def __init__(self, Nx, Nw, Ny):
        super(SinkBF_moving_average, self).__init__()
        self.Nx = Nx
        self.Nw = Nw
        self.Ny = Ny

        self._alpha = nn.Parameter(torch.rand(Nx))

        self.q_x_y = nn.Sequential(nn.Linear(Ny, Nx), nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.q_w_xy = nn.Sequential(nn.Linear(Nx+Ny, Nw), nn.Tanh(),
            nn.Linear(Nw, Nw),)
        self.f_x_xw = nn.Sequential(nn.Linear(Nx+Nw, Nx), nn.Tanh(),
            nn.Linear(Nx, Nx),)

    def forward(self, _Y0, Nhrzn):
# _Y0: (Nwup+1, *, Ny)
        Nx, Nw = self.Nx, self.Nw

        assert _Y0.shape[0] >= 1, "Y0 is not allowed to be empty."
        Nwup = _Y0.shape[0] - 1
        Nbatch = _Y0.shape[1]

        _alpha = torch.max(torch.min(self._alpha, torch.ones(Nx)), 
            -torch.ones(Nx)) # (Nx,)
        _beta = torch.sqrt(1-_alpha**2) # (Nx,)

        _x = self.q_x_y(_Y0[0,:]) # (*, Nx)
        X = [_x,]
        W = []
        for t in range(1, Nwup+1+Nhrzn):
            if t <= Nwup:
                _w = self.q_w_xy(torch.cat((_x, _Y0[t,:]), dim=1)) # (*, Nw)
                W.append(_w)
            else:
                if self.training:
                    _w = torch.zeros(Nbatch, Nw)
                else:
                    _w = torch.randn(Nbatch, Nw)
            #_x = _alpha * _x \
            #    + _beta * self.f_x_xw(torch.cat((_x, _w), dim=1)) # (*, Nx)
            _x = _x  + self.f_x_xw(torch.cat((_x, _w), dim=1)) # (*, Nx)

            X.append(_x)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        if len(W) > 0:
            _W = torch.stack(W, dim = 0) # (Nwup, *, Nw)
        else:
            _W = torch.zeros(Nwup, Nbatch, Nw)
        _Yhat = self.p_y_x(_X) # (Nwup+1+Nhrzn, *, Ny)

        return _X, _W, _Yhat


class SinkBF_uniform(nn.Module):
    def __init__(self, Nx, Nw, Ny):
        super(SinkBF_uniform, self).__init__()
        self.Nx = Nx
        self.Nw = Nw
        self.Ny = Ny

        self.q_x_y = nn.Sequential(nn.Linear(Ny, Nx), nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.q_w_xy = nn.Sequential(nn.Linear(Nx+Ny, Nw), nn.Tanh(),)
        self.f_x_xw = nn.GRU(Nw, Nx, 1) # num_layers = 1

    def forward(self, _Y0, Nhrzn):
# _Y0: (Nwup+1, *, Ny)
        Nx, Nw = self.Nx, self.Nw

        assert _Y0.shape[0] >= 1, "Y0 is not allowed to be empty."
        Nwup = _Y0.shape[0] - 1
        Nbatch = _Y0.shape[1]

        _x = self.q_x_y(_Y0[0,:]) # (*, Nx)
        X = [_x,]
        W = []
        for t in range(1, Nwup+1+Nhrzn):
            if t <= Nwup:
                _w = self.q_w_xy(torch.cat((_x, _Y0[t,:]), dim=1)) # (*, Nw)
                W.append(_w)
            else:
                if self.training:
                    _w = torch.zeros(Nbatch, Nw)
                else:
                    _w = 2*torch.rand(Nbatch, Nw)-1
            _x = self.f_x_xw(_w.unsqueeze(0), _x.unsqueeze(0))[1][0,:] # (*, Nx)
            X.append(_x)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        if len(W) > 0:
            _W = torch.stack(W, dim = 0) # (Nwup, *, Nw)
        else:
            _W = torch.zeros(Nwup, Nbatch, Nw)
        _Yhat = self.p_y_x(_X) # (Nwup+1+Nhrzn, *, Ny)

        return _X, _W, _Yhat

class SinkBF_multilayers(nn.Module):
    def __init__(self, Nx, Nw, Ny, Nlayer):
        super(SinkBF_multilayers, self).__init__()
        self.Nx = Nx
        self.Nw = Nw
        self.Ny = Ny
        self.Nlayer = Nlayer

        self.q_x_y = nn.Sequential(nn.Linear(Ny, Nx*Nlayer), nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.q_w_xy = nn.Sequential(nn.Linear(Nx+Ny, Nw), nn.Tanh(),)
        self.f_x_xw = nn.GRU(Nw, Nx, Nlayer) 

    def forward(self, _Y0, Nhrzn):
# _Y0: (Nwup+1, *, Ny)
        Nx, Nw, Nlayer = self.Nx, self.Nw, self.Nlayer

        assert _Y0.shape[0] >= 1, "Y0 is not allowed to be empty."
        Nwup = _Y0.shape[0] - 1
        Nbatch = _Y0.shape[1]

        _x = self.q_x_y(_Y0[0,:]).reshape(Nbatch, Nlayer, Nx) # (*, Nlayer, Nx)
        _x = _x.transpose(0,1) # (Nlayer, *, Nx)
        _yhat = self.p_y_x(_x[-1,:]) # (*, Ny)
        X = [_x[-1,:],]
        Yhat = [_yhat,]
        W = []
        for t in range(1, Nwup+1+Nhrzn):
            if t <= Nwup:
                _e = _Y0[t,:] - _yhat # (*, Ny)
                _w = self.q_w_xy(torch.cat((_x[0,:], _e), dim=1)) # (*, Nw)
                W.append(_w)
            else:
                if self.training:
                    _w = torch.zeros(Nbatch, Nw)
                else:
                    _w = 2*torch.rand(Nbatch, Nw)-1
            _, _x = self.f_x_xw(_w.unsqueeze(0), _x) # (Nlayer, *, Nx)
            _yhat = self.p_y_x(_x[-1,:]) # (*, Ny)
            X.append(_x[-1,:])
            Yhat.append(_yhat)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        _Yhat = torch.stack(Yhat, dim = 0) # (Nwup+1+Nhrzn, *, Ny)
        if len(W) > 0:
            _W = torch.stack(W, dim = 0) # (Nwup, *, Nw)
        else:
            _W = torch.zeros(Nwup, Nbatch, Nw)

        return _X, _W, _Yhat

def run_training_004(sbf_guass, data_generator, optimizer, Nepoch, Nbatch, \
    Nwup, Nhrzn, reg_param, eps_given = 1e-1, tol = 1e-2, max_itr = 2**10, 
    loss_hist = None, rand = 'gauss'):

    if loss_hist is None:
        loss_hist = []

    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch
    Nx, Nw, Ny = sbf_guass.Nx, sbf_guass.Nw, sbf_guass.Ny

    t_bgn = datetime.now()
    for epoch in range(Nepoch):
        tmp = []
        for k1 in range(Nitr):
            Ybatch = data_generator.batch(Nbatch, Nwup+1+Nhrzn)
            _Y0batch = torch.tensor(Ybatch[:Nwup+1,:]) # (Nwup+1, *, Ny)
            _Ybatch = torch.tensor(Ybatch) # (Nwup+1+Nhrzn, *, Ny)

            _Xbatch, _W, _Yhat_batch = sbf_guass(_Y0batch, Nhrzn)
# _Xbatch: (Nwup+1+Nhrzn, *, Nx), _W: (Nwup, *, Nw), 
# _Yhat_batch: (Nwup+1+Nhrzn, *, Ny)
            _resid = torch.mean((_Yhat_batch - _Ybatch)**2)

            if Nwup > 1:

                idx0a, idx0b = np.random.permutation(Nwup)[0:2]
                idx0a, idx0b = torch.tensor(idx0a), torch.tensor(idx0b)

                idx0 = torch.randint(0, Nwup, size= (()))
                idx1 = torch.randint(0, Nbatch, size = (()))
                idx2 = torch.randint(0, Nw, size = (()))

                _testW0 = _W[idx0, :, idx2].reshape(-1,1) # (Nbatch, 1)
                _testW1 = (_W[idx0a, :, idx2] - _W[idx0b, :, idx2])/torch.tensor(np.sqrt(2)).reshape(-1,1) # (Nbatch, 1)

                if rand == "gauss":
                    _refW0 = torch.randn(2**7, 1) # (2**7, 1)
                    _refW1 = torch.randn(2**7, 1) # (2**7, 1)

                if rand == "uniform":
                    _refW0 = 2*torch.rand(2**7, 1)-1 # (2**7, 1)

                    _tmp1 = 2*torch.rand(2**7, 1)-1 # (2**7, 1)
                    _tmp2 = 2*torch.rand(2**7, 1)-1 # (2**7, 1)

                    _refW1 = (_tmp1-_tmp2)/torch.tensor(np.sqrt(2)) # (2**7, 1)

                _dist0, cnt = measure_distance(_testW0, _refW0, tol = tol, 
                    eps_given = eps_given, max_itr = max_itr, norm_p = 2)

                _dist1, cnt = measure_distance(_testW1, _refW1, tol = tol, 
                    eps_given = eps_given, max_itr = max_itr, norm_p = 2)

                _dist = (_dist0 + _dist1)/2
            else:
                _dist = torch.zeros(())

            _loss = _resid + reg_param * _dist
            tmp.append((float(_loss), float(_resid), float(_dist)))

            sbf_guass.zero_grad()
            _loss.backward()
            optimizer.step()

        loss_avg, resid_avg, dist_avg = np.mean(tmp, axis=0)
        elapsed_time = datetime.now() - t_bgn
        sys.stdout.write('%s epoch %04d loss %8.2e resid %8.2e dist %8.2e\r' %
            (elapsed_time, epoch, loss_avg, resid_avg, dist_avg))

        loss_hist.append((elapsed_time, epoch, loss_avg, resid_avg, dist_avg,))

    return loss_hist

