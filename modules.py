
import torch
import torch.nn as nn
import numpy as np
import sys
from datetime import datetime, timedelta


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
                _w = torch.randn(Nbatch, Nw)
            _x = self.f_x_xw(_w.unsqueeze(0), _x.unsqueeze(0))[1][0,:] # (*, Nx)
            X.append(_x)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        _W = torch.stack(W, dim = 0) # (Nwup, *, Nw)
        _Yhat = self.p_y_x(_X) # (Nwup+1+Nhrzn, *, Ny)

        return _X, _W, _Yhat
        


def robust_sinkhorn_iteration(_M, _p, _q, _eps, tol, max_itr):
    _alpha = _p * 0
    _beta = _q * 0
    cnt = 0
    
    assert max(_M.shape) <= 2**6, "The shape of M is %s. That exceeds the limitation: 64" % str(_M.shape)

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

def measure_distance(_X0, _X1, tol = 1e-4, eps_given = 1e-2, max_itr = 2**5):
    # _X0: (*, Nx), _X1: (*, Nx)
        
    _M01 = torch.mean((_X0.unsqueeze(1) - _X1)**2, dim=2)

    _p = 1/_X0.shape[0] * torch.ones(_X0.shape[0])
    _q = 1/_X1.shape[0] * torch.ones(_X1.shape[0])

    _eps = torch.tensor(eps_given) * float(torch.mean(_M01))
        
    _dist01, cnt01 = robust_sinkhorn_iteration(_M01, _p.unsqueeze(1), 
        _q.unsqueeze(0), _eps, tol, max_itr=max_itr)
    
    return _dist01, cnt01


def run_training(sbf, data_generator, optimizer, Nepoch, Nbatch, Nwup, Nhrzn,
    reg_param):
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
            _reg_term, _ = measure_distance(_X0, torch.randn(Nbatch, Nx))

            for k1 in range(Nwup):
                _reg_term += measure_distance(_X0, torch.randn(Nbatch, Nx))[0]
            _reg_term /= (Nwup+1)
            
            _loss = _resid + reg_param * _reg_term
            loss_hist.append((float(_resid), float(_reg_term), float(_loss)))

            sbf.zero_grad()
            _loss.backward()
            optimizer.step()

        resid_avg, reg_term_avg, loss_avg = np.mean(loss_hist, axis=0)
        sys.stdout.write('%s epoch %04d resid %8.2e reg term %8.2e loss %8.2e\n' %
            (datetime.now() - t_bgn, epoch+1, resid_avg, reg_term_avg, loss_avg))
    

class SinkBF_binary(nn.Module):
    def __init__(self, Nx, Nw, Ny):
        super(SinkBF_binary, self).__init__()
        self.Nx = Nx
        self.Nw = Nw
        self.Ny = Ny

        self.q_x_y = nn.Sequential(nn.Linear(Ny, Nx), nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.logit_q_w_xy = nn.Sequential(nn.Linear(Nx+Ny, Nw), nn.Tanh(),
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
        logit_Q_w = []
        for t in range(1, Nwup+1+Nhrzn):
            if t <= Nwup:
                _logit_q_w = self.logit_q_w_xy(torch.cat((_x, _Y0[t,:]), \
                    dim=1)) # (*, Nw)
                _w = torch.tanh(_logit_q_w/2) # (*, Nw)
                logit_Q_w.append(_logit_q_w)
            else:
                _w = torch.ones(Nbatch, Nw)
                _w[torch.rand(Nbatch, Nw) >= 0.5] = -1
            _x = self.f_x_xw(_w.unsqueeze(0), _x.unsqueeze(0))[1][0,:] # (*, Nx)
            X.append(_x)
        _X = torch.stack(X, dim = 0) # (Nwup+1+Nhrzn, *, Nx)
        _logit_Q_w = torch.stack(logit_Q_w, dim = 0) # (Nwup, *, Nw)
        _Yhat = self.p_y_x(_X) # (Nwup+1+Nhrzn, *, Ny)

        return _X, _logit_Q_w, _Yhat


def run_training_binary(sbf_binary, data_generator, optimizer, Nepoch, Nbatch, \
    Nwup, Nhrzn, reg_param):
    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch
    Nx, Nw, Ny = sbf_binary.Nx, sbf_binary.Nw, sbf_binary.Ny

    t_bgn = datetime.now()
    for epoch in range(Nepoch):
        loss_hist = []
        for k1 in range(Nitr):
            Ybatch = data_generator.batch(Nbatch, Nwup+1+Nhrzn)
            _Y0batch = torch.tensor(Ybatch[:Nwup+1,:]) # (Nwup+1, *, Ny)
            _Ybatch = torch.tensor(Ybatch) # (Nwup+1+Nhrzn, *, Ny)

            _Xbatch, _logit_Q_w_batch, _Yhat_batch = sbf_binary(_Y0batch, Nhrzn)
            _resid = torch.mean((_Yhat_batch - _Ybatch)**2)

            _discrepancy_q = torch.mean(torch.tanh(torch.abs(_logit_Q_w_batch)/2))

            _loss = _resid + reg_param * _discrepancy_q
            loss_hist.append((float(_resid), float(_discrepancy_q), 
                float(_loss)))

            sbf_binary.zero_grad()
            _loss.backward()
            optimizer.step()

        resid_avg, reg_term_avg, loss_avg = np.mean(loss_hist, axis=0)
        sys.stdout.write('%s epoch %04d resid %8.2e reg term %8.2e loss %8.2e\r' %
            (datetime.now() - t_bgn, epoch+1, resid_avg, reg_term_avg, loss_avg))
