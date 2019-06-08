
import unittest
from modules import *
from data import *
import matplotlib.pylab as plt

class TestCases(unittest.TestCase):
    
    def test_001(self):
        Nx, Nw, Ny = 2**3, 2**2, 2**4
        sbf = SinkBF_gauss(Nx, Nw, Ny)

    def test_002(self):
        Nx, Nw, Ny = 2**3, 2**2, 2**4
        sbf = SinkBF_gauss(Nx, Nw, Ny)

        Nwup, Nhrzn, Nbatch = 2**3, 2**4, 2**5

        _Y0 = torch.randn(Nwup+1, Nbatch, Ny)
        _X, _W, _Yhat = sbf(_Y0, Nhrzn)

        assert _X.shape == (Nwup+1+Nhrzn, Nbatch, Nx)
        assert _W.shape == (Nwup, Nbatch, Nw)
        assert _Yhat.shape == (Nwup+1+Nhrzn, Nbatch, Ny)

    def test_003(self):
        generator = GeneratorFromLorenzAttractor()

        Nbatch, N = 2**3, 2**6
        Xbatch = generator.batch(Nbatch, N)

        assert Xbatch[0,0,0].dtype == np.float32
        assert Xbatch.shape == (N, Nbatch, 3)

        #plt.plot(Xbatch[:,0,:])
        #plt.show()

        Xbatch, Nbatch = generator.test(N)
        assert Xbatch[0,0,0].dtype == np.float32
        assert Xbatch.shape == (N, Nbatch, 3)

        #plt.plot(Xbatch[:,0,:])
        #plt.show()

    def test_004(self):

        Nbatch, Nx = 2**5, 2**3

        for k1 in range(10):
            _X0 = torch.randn(Nbatch, Nx) 
            _X1 = torch.randn(Nbatch, Nx)
            _dist, cnt = measure_distance(_X0, _X1, eps_given = 1e-2, \
                max_itr = 2**5)

    def test_005(self):
        Nx, Nw, Ny = 2**3, 2**3, 3
        sbf = SinkBF_gauss(Nx, Nw, Ny)

        optimizer = torch.optim.Adam(sbf.parameters())

        data_generator = GeneratorFromLorenzAttractor()

        Nepoch, Nbatch, Nwup, Nhrzn = 2**0, 2**6, 2**3, 2**6
        reg_param = 0.1

        run_training(sbf, data_generator, optimizer, Nepoch, Nbatch,
            Nwup, Nhrzn, reg_param)

        
if __name__ == "__main__":
    unittest.main()


