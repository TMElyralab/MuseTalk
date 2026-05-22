#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


__all__ = ['inv', 'svd', 'det', 'slogdet', 'pinv', 'lstsq', 'norm']

import numpy as np
import scipy.sparse as sp
from .ch import Ch, depends_on
from .ch_ops import NanDivide
from .ch_ops import asarray as ch_asarray
from .ch_ops import sqrt as ch_sqrt
from .ch_ops import sum as ch_sum
from .reordering import concatenate as ch_concatenate
from .ch_random import randn as ch_random_randn
from .utils import row, col


try:
    asarray = ch_asarray
    import inspect
    exec(''.join(inspect.getsourcelines(np.linalg.tensorinv)[0]))
    __all__.append('tensorinv')
except: pass

def norm(x, ord=None, axis=None):
    if ord is not None or axis is not None:
        raise NotImplementedError("'ord' and 'axis' should be None for now.")

    return ch_sqrt(ch_sum(x**2))

# This version works but derivatives are too slow b/c of nested loop in Svd implementation.
# def lstsq(a, b):
#     u, s, v = Svd(a)
#     x = (v.T / s).dot(u.T.dot(b))
#     residuals = NotImplementedError # ch_sum((a.dot(x) - b)**2, axis=0)
#     rank = NotImplementedError
#     s = NotImplementedError
#     return x, residuals, rank, s

def lstsq(a, b, rcond=-1):
    if rcond != -1:
        raise Exception('non-default rcond not yet implemented')
        
    x = Ch(lambda a, b : pinv(a).dot(b))
    x.a = a
    x.b = b
    residuals = ch_sum(  (x.a.dot(x) - x.b) **2 , axis=0)
    rank = NotImplementedError
    s = NotImplementedError
    
    return x, residuals, rank, s

def Svd(x, full_matrices=0, compute_uv=1):
    
    if full_matrices != 0:
        raise Exception('full_matrices must be 0')
    if compute_uv != 1:
        raise Exception('compute_uv must be 1')
        
    need_transpose = x.shape[0] < x.shape[1]
    
    if need_transpose:
        x = x.T
        
    svd_d = SvdD(x=x)
    svd_v = SvdV(x=x, svd_d=svd_d)
    svd_u = SvdU(x=x, svd_d=svd_d, svd_v=svd_v)

    if need_transpose:
        return svd_v, svd_d, svd_u.T
    else:
        return svd_u, svd_d, svd_v.T

    
class Pinv(Ch):
    dterms = 'mtx'
    
    def on_changed(self, which):
        mtx = self.mtx
        if mtx.shape[1] > mtx.shape[0]:
            result = mtx.T.dot(Inv(mtx.dot(mtx.T)))
        else:
            result = Inv(mtx.T.dot(mtx)).dot(mtx.T)
        self._result = result
        
    def compute_r(self):
        return self._result.r
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.mtx:
            return self._result.dr_wrt(self.mtx)
        
# Couldn't make the SVD version of pinv work yet...
#
# class Pinv(Ch):
#     dterms = 'mtx'
#     
#     def on_changed(self, which):
#         u, s, v = Svd(self.mtx)
#         result = (v.T * (NanDivide(1.,row(s)))).dot(u.T)
#         self.add_dterm('_result', result)
# 
#     def compute_r(self):
#         return self._result.r
#         
#     def compute_dr_wrt(self, wrt):
#         if wrt is self._result:
#             return 1



class LogAbsDet(Ch):
    dterms = 'x'
    
    def on_changed(self, which):
        self.sign, self.slogdet = np.linalg.slogdet(self.x.r)
    
    def compute_r(self):
        return self.slogdet
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:         
            return row(np.linalg.inv(self.x.r).T)

class SignLogAbsDet(Ch):
    dterms = 'logabsdet',
    
    def compute_r(self):
        _ = self.logabsdet.r
        return self.logabsdet.sign
        
    def compute_dr_wrt(self, wrt):
        return None


class Det(Ch):
    dterms = 'x'

    def compute_r(self):
        return np.linalg.det(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return row(self.r * np.linalg.inv(self.x.r).T)


class Inv(Ch):
    dterms = 'a'
    
    def compute_r(self):
        return np.linalg.inv(self.a.r)
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a:
            return None
    
        Ainv = self.r

        if Ainv.ndim <= 2:
            return -np.kron(Ainv, Ainv.T)
        else:
            Ainv = np.reshape(Ainv,  (-1, Ainv.shape[-2], Ainv.shape[-1]))
            AinvT = np.rollaxis(Ainv, -1, -2)
            AinvT = np.reshape(AinvT, (-1, AinvT.shape[-2], AinvT.shape[-1]))
            result = np.dstack([-np.kron(Ainv[i], AinvT[i]).T for i in range(Ainv.shape[0])]).T
            result = sp.block_diag(result)

        return result


class SvdD(Ch):
    dterms = 'x'

    @depends_on('x')
    def UDV(self):
        result = np.linalg.svd(self.x.r, full_matrices=False)
        result = [result[0], result[1], result[2].T]
        result[1][np.abs(result[1]) < np.spacing(1)] = 0.
        return result
    
    def compute_r(self):
        return self.UDV[1]
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        
        u, d, v = self.UDV
        shp = self.x.r.shape
        u = u[:shp[0], :shp[1]]
        v = v[:shp[1], :d.size]
        
        result = np.einsum('ik,jk->kij', u, v)
        result = result.reshape((result.shape[0], -1))
        return result
    
    
class SvdV(Ch):
    terms = 'svd_d'
    dterms = 'x'
    
    def compute_r(self):
        return self.svd_d.UDV[2]

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        
        U,_D,V = self.svd_d.UDV
        
        shp = self.svd_d.x.r.shape
        mxsz = max(shp[0], shp[1])
        #mnsz = min(shp[0], shp[1])
        D = np.zeros(mxsz)
        D[:_D.size] = _D

        omega = np.zeros((shp[0], shp[1], shp[1], shp[1]))

        M = shp[0]
        N = shp[1]
        
        assert(M >= N)
        
        for i in range(shp[0]):
            for j in range(shp[1]):
                for k in range(N):
                    for l in range(k+1, N):
                        mtx = np.array([
                            [D[l],D[k]],
                            [D[k],D[l]]])
                    
                        rhs = np.array([U[i,k]*V[j,l], -U[i,l]*V[j,k]])
                        result = np.linalg.solve(mtx, rhs)
                        
                        omega[i,j,k,l] =  result[1]
                        omega[i,j,l,k] = -result[1]
                   
        #print 'v size is %s' % (str(V.shape),)
        #print 'v omega size is %s' % (str(omega.shape),)
        assert(V.shape[1] == omega.shape[2])
        return np.einsum('ak,ijkl->alij', -V, omega).reshape((self.r.size, wrt.r.size))
        
        
class SvdU(Ch):
    dterms = 'x'
    terms = 'svd_d', 'svd_v'

    def compute_r(self):
        return self.svd_d.UDV[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            # return (
            #     self.svd_d.x.dot(self.svd_v)
            #     /
            #     self.svd_d.reshape((1,-1))
            #     ).dr_wrt(self.svd_d.x)
            return (
                NanDivide(
                    self.svd_d.x.dot(self.svd_v),
                    self.svd_d.reshape((1,-1)))
                ).dr_wrt(self.svd_d.x)
    

inv = Inv
svd = Svd
det = Det
pinv = Pinv

def slogdet(*args):
    n = len(args)
    if n == 1:
        r2 = LogAbsDet(x=args[0])
        r1 = SignLogAbsDet(r2)
        return r1, r2
    else:
        r2 = [LogAbsDet(x=arg) for arg in args]
        r1 = [SignLogAbsDet(r) for r in r2]
        r2 = ch_concatenate(r2)
        return r1, r2

def main():
    
    tmp = ch_random_randn(100).reshape((10,10))
    print('chumpy version: ' + str(slogdet(tmp)[1].r))
    print('old version:' + str(np.linalg.slogdet(tmp.r)[1]))

    eps = 1e-10
    diff = np.random.rand(100) * eps
    diff_reshaped = diff.reshape((10,10))
    print(np.linalg.slogdet(tmp.r+diff_reshaped)[1] - np.linalg.slogdet(tmp.r)[1])
    print(slogdet(tmp)[1].dr_wrt(tmp).dot(diff))
    
    print(np.linalg.slogdet(tmp.r)[0])
    print(slogdet(tmp)[0])

if __name__ == '__main__':
    main()

