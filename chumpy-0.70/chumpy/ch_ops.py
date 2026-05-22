#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

# Numpy functions
__all__ = ['array', 'amax','amin', 'max', 'min', 'maximum','minimum','nanmax','nanmin',
            'sum', 'exp', 'log', 'mean','std', 'var',
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
            'sqrt', 'square', 'absolute', 'abs', 'clip',
            'power',
            'add', 'divide', 'multiply', 'negative', 'subtract', 'reciprocal',
            'nan_to_num',
            'dot', 'cumsum',
            'floor', 'ceil',
            'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal',
            'nonzero', 'ascontiguousarray', 'asfarray', 'arange', 'asarray', 'copy',
            'cross',
            'shape', 'sign']


__all__ += ['SumOfSquares',
           'NanDivide', ]
           

# These can be wrapped directly as Ch(routine(*args, **kwargs)),
# so that for example "ch.eye(3)" translates into Ch(np.eye(3))
numpy_array_creation_routines = [
    'empty','empty_like','eye','identity','ones','ones_like','zeros','zeros_like',
    'array',
    'arange','linspace','logspace','meshgrid','mgrid','ogrid',
    'fromfunction', 'fromiter', 'meshgrid', 'tri'
]


wont_implement = ['asanyarray', 'asmatrix', 'frombuffer', 'copy', 'fromfile', 'fromstring', 'loadtxt', 'copyto', 'asmatrix', 'asfortranarray', 'asscalar', 'require']
not_yet_implemented = ['tril', 'triu', 'vander']

__all__ += not_yet_implemented
__all__ += wont_implement
__all__ += numpy_array_creation_routines
    
    
from .ch import Ch
import six
import numpy as np
import warnings
from six.moves import cPickle as pickle
import scipy.sparse as sp
from .utils import row, col
from copy import copy as copy_copy
from functools import reduce

__all__ += ['pi', 'set_printoptions']
pi = np.pi
set_printoptions = np.set_printoptions
arange = np.arange

for rtn in ['argmax', 'nanargmax', 'argmin', 'nanargmin']:
    exec('def %s(a, axis=None) : return np.%s(a.r, axis) if hasattr(a, "compute_r") else np.%s(a, axis)' % (rtn, rtn, rtn))
    __all__ += [rtn]
    
for rtn in ['argwhere', 'nonzero', 'flatnonzero']:
    exec('def %s(a) : return np.%s(a.r) if hasattr(a, "compute_r") else np.%s(a)' % (rtn, rtn, rtn))
    __all__ += [rtn]

for rtn in numpy_array_creation_routines:
    exec('def %s(*args, **kwargs) : return Ch(np.%s(*args, **kwargs))' % (rtn, rtn))


class WontImplement(Exception):
    pass

for rtn in wont_implement:
    exec('def %s(*args, **kwargs) : raise WontImplement' % (rtn))
    
for rtn in not_yet_implemented:
    exec('def %s(*args, **kwargs) : raise NotImplementedError' % (rtn))

def asarray(a, dtype=None, order=None):
    assert(dtype is None or dtype is np.float64)
    assert(order is 'C' or order is None)
    if hasattr(a, 'dterms'):
        return a
    return Ch(np.asarray(a, dtype, order))

# Everythign is always c-contiguous
def ascontiguousarray(a, dtype=None): return a

# Everything is always float
asfarray = ascontiguousarray

def copy(self):
    return pickle.loads(pickle.dumps(self))    

def asfortranarray(a, dtype=None): raise WontImplement


class Simpleton(Ch):
    dterms = 'x'    
    def compute_dr_wrt(self, wrt): 
        return None

class floor(Simpleton):
    def compute_r(self): return np.floor(self.x.r)

class ceil(Simpleton):
    def compute_r(self): return np.ceil(self.x.r)

class sign(Simpleton):
    def compute_r(self): return np.sign(self.x.r)

class Cross(Ch):
    dterms = 'a', 'b'
    terms = 'axisa', 'axisb', 'axisc', 'axis'
    term_order = 'a', 'b', 'axisa', 'axisb', 'axisc', 'axis'
    
    def compute_r(self):
        return np.cross(self.a.r, self.b.r, self.axisa, self.axisb, self.axisc, self.axis)
        
    
    def _load_crossprod_cache(self, h, w):
        if not hasattr(self, '_w'):
            self._w = 0
            self._h = 0

        if h!=self._h or w!=self._w:
            sz = h*w
            rng = np.arange(sz)
            self._JS = np.repeat(rng.reshape((-1,w)), w, axis=0).ravel()
            self._IS = np.repeat(rng, w)
            self._tiled_identity = np.tile(np.eye(w), (h, 1))
            self._h = h
            self._w = w
            
        return self._tiled_identity, self._IS, self._JS, 
            
    

    # Could be at least 2x faster, with some work
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return

        sz = self.a.size
        h, w = self.a.shape
        tiled_identity, IS, JS = self._load_crossprod_cache(h, w)
        
        #import time
        #tm = time.time()
        if wrt is self.a:
            rp = np.repeat(-self.b.r, w, axis=0) 
            result = np.cross(
                tiled_identity, 
                rp,
                self.axisa,
                self.axisb, 
                self.axisc, 
                self.axis)

        elif wrt is self.b:
            result = np.cross(
                np.repeat(-self.a.r, w, axis=0),
                tiled_identity,
                self.axisa,
                self.axisb, 
                self.axisc, 
                self.axis)
                
        # rng = np.arange(sz)
        # JS = np.repeat(rng.reshape((-1,w)), w, axis=0).ravel()
        # IS = np.repeat(rng, w)
        data = result.ravel()
        result = sp.csc_matrix((data, (IS,JS)), shape=(self.size, wrt.size))
        #import pdb; pdb.set_trace()
        #print 'B TOOK %es' % (time.time() -tm )
        return result
    
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return Cross(a, b, axisa, axisb, axisc, axis)
    



class cumsum(Ch):
    dterms = 'a'
    terms = 'axis'
    term_order = 'a', 'axis'
    
    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
    
    def compute_r(self):
        return np.cumsum(self.a.r, axis=self.axis)
        
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a:
            return None
        
        if self.axis is not None:
            raise NotImplementedError
            
        IS = np.tile(row(np.arange(self.a.size)), (self.a.size, 1))
        JS = IS.T
        IS = IS.ravel()
        JS = JS.ravel()
        which = IS >= JS
        IS = IS[which]
        JS = JS[which]
        data = np.ones_like(IS)
        result = sp.csc_matrix((data, (IS, JS)), shape=(self.a.size, self.a.size))
        return result
            

class UnaryElemwise(Ch):
    dterms = 'x'
    
    def compute_r(self):
        return self._r(self.x.r)
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            result = self._d(self.x.r)
            return sp.diags([result.ravel()], [0]) if len(result)>1 else np.atleast_2d(result)


class nan_to_num(UnaryElemwise):
    _r = lambda self, x : np.nan_to_num(x)
    _d = lambda self, x : np.asarray(np.isfinite(x), np.float64)

class reciprocal(UnaryElemwise):
    _r = np.reciprocal
    _d = lambda self, x : -np.reciprocal(np.square(x))

class square(UnaryElemwise):
    _r = np.square
    _d = lambda self, x : x * 2.

def my_power(a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        return np.nan_to_num(np.power(a, b))

class sqrt(UnaryElemwise):
    _r = np.sqrt
    _d = lambda self, x : .5 * my_power(x, -0.5)

class exp(UnaryElemwise):
    _r = np.exp
    _d = np.exp    

class log(UnaryElemwise):
    _r = np.log
    _d = np.reciprocal

class sin(UnaryElemwise):
    _r = np.sin
    _d = np.cos
    
class arcsin(UnaryElemwise):
    _r = np.arcsin
    _d = lambda self, x : np.reciprocal(np.sqrt(1.-np.square(x)))
    
class cos(UnaryElemwise):
    _r = np.cos
    _d = lambda self, x : -np.sin(x)

class arccos(UnaryElemwise):
    _r = np.arccos
    _d = lambda self, x : -np.reciprocal(np.sqrt(1.-np.square(x)))

class tan(UnaryElemwise):
    _r = np.tan
    _d = lambda self, x : np.reciprocal(np.cos(x)**2.)
    
class arctan(UnaryElemwise):
    _r = np.arctan
    _d = lambda self, x : np.reciprocal(np.square(x)+1.)
    
class negative(UnaryElemwise):
    _r = np.negative
    _d = lambda self, x : np.negative(np.ones_like(x))

class absolute(UnaryElemwise):
    _r = np.abs
    _d = lambda self, x : (x>0)*2-1.

abs = absolute

class clip(Ch):
    dterms = 'a'
    terms = 'a_min', 'a_max'
    term_order = 'a', 'a_min', 'a_max'
    
    def compute_r(self):
        return np.clip(self.a.r, self.a_min, self.a_max)
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.a:
            result = np.asarray((self.r != self.a_min) & (self.r != self.a_max), np.float64)
            return sp.diags([result.ravel()], [0]) if len(result)>1 else np.atleast_2d(result)

class sum(Ch):
    dterms = 'x',
    terms  = 'axis',
    term_order = 'x', 'axis'
    
    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
        if not hasattr(self, 'dr_cache'):
            self.dr_cache = {}

    def compute_r(self):
        return np.sum(self.x.r, axis=self.axis)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        if self.axis == None:
            return row(np.ones((1, len(self.x.r.ravel()))))
        else:
            uid = tuple(list(self.x.shape) + [self.axis])
            if uid not in self.dr_cache:
                idxs_presum = np.arange(self.x.size).reshape(self.x.shape)
                idxs_presum = np.rollaxis(idxs_presum, self.axis, 0)
                idxs_postsum = np.arange(self.r.size).reshape(self.r.shape)
                tp = np.ones(idxs_presum.ndim, dtype=np.uint32)
                tp[0] = idxs_presum.shape[0]
                idxs_postsum = np.tile(idxs_postsum, tp)
                data = np.ones(idxs_postsum.size)
                result = sp.csc_matrix((data, (idxs_postsum.ravel(), idxs_presum.ravel())), (self.r.size, wrt.size))
                self.dr_cache[uid] = result
            return self.dr_cache[uid]
            

class mean(Ch):
    dterms = 'x',
    terms  = 'axis',
    term_order = 'x', 'axis'
    
    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
        if not hasattr(self, 'dr_cache'):
            self.dr_cache = {}

    def compute_r(self):
        return np.array(np.mean(self.x.r, axis=self.axis))

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        if self.axis == None:
            return row(np.ones((1, len(self.x.r))))/len(self.x.r)
        else:
            uid = tuple(list(self.x.shape) + [self.axis])
            if uid not in self.dr_cache:
                idxs_presum = np.arange(self.x.size).reshape(self.x.shape)
                idxs_presum = np.rollaxis(idxs_presum, self.axis, 0)
                idxs_postsum = np.arange(self.r.size).reshape(self.r.shape)
                tp = np.ones(idxs_presum.ndim, dtype=np.uint32)
                tp[0] = idxs_presum.shape[0]
                idxs_postsum = np.tile(idxs_postsum, tp)
                data = np.ones(idxs_postsum.size) / self.x.shape[self.axis]
                result = sp.csc_matrix((data, (idxs_postsum.ravel(), idxs_presum.ravel())), (self.r.size, wrt.size))
                self.dr_cache[uid] = result
            return self.dr_cache[uid]


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if (dtype != None or out != None or ddof != 0 or keepdims != False):
        raise NotImplementedException('Unimplemented for non-default dtype, out, ddof, and keepdims.')
    return mean(a**2., axis=axis)
    
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if (dtype != None or out != None or ddof != 0 or keepdims != False):
        raise NotImplementedException('Unimplemented for non-default dtype, out, ddof, and keepdims.')
    return sqrt(var(a, axis=axis))
    

class SumOfSquares(Ch):
    dterms = 'x',

    def compute_r(self):
        return np.sum(self.x.r.ravel()**2.)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return row(self.x.r.ravel()*2.)
    
    
class divide (Ch):
    dterms = 'x1', 'x2'

    def compute_r(self):
        return self.x1.r / self.x2.r

    def compute_dr_wrt(self, wrt):
        
        if (wrt is self.x1) == (wrt is self.x2):
            return None
            
        IS, JS, input_sz, output_sz = _broadcast_setup(self.x1, self.x2, wrt)
        
        x1r, x2r = self.x1.r, self.x2.r
        if wrt is self.x1:
            data = (np.ones_like(x1r) / x2r).ravel()
        else:
            data = (-x1r / (x2r*x2r)).ravel()
            
        return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))


            

class NanDivide(divide):
    dterms = 'x1', 'x2'
    
    def compute_r(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = super(self.__class__, self).compute_r()
        shape = result.shape
        result = result.ravel()
        result[np.isinf(result)] = 0
        result[np.isnan(result)] = 0
        return result.reshape(shape)
        
    def compute_dr_wrt(self, wrt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = super(self.__class__, self).compute_dr_wrt(wrt)        
        if result is not None:
            result = result.copy()
            if sp.issparse(result):
                result.data[np.isinf(result.data)] = 0
                result.data[np.isnan(result.data)] = 0
                return result
            else:
                rr = result.ravel()
                rr[np.isnan(rr)] = 0.
                rr[np.isinf(rr)] = 0.
                return result


def shape(a):
    return a.shape if hasattr(a, 'shape') else np.shape(a)    


_bs_setup_data1 = {}
_bs_setup_data2 = {}
def _broadcast_matrix(a, b, wrt, data):
    global _bs_setup_data1, _bs_setup_data2

    if len(set((a.shape, b.shape))) == 1:
        uid = a.shape
        if uid not in _bs_setup_data1:
            asz = a.size
            IS = np.arange(asz)
            _bs_setup_data1[uid] = sp.csc_matrix((np.empty(asz), (IS, IS)), shape=(asz, asz))
        result = copy_copy(_bs_setup_data1[uid])
        if isinstance(data, np.ndarray):
            result.data = data.ravel()
        else: # assumed scalar
            result.data = np.empty(result.nnz)
            result.data.fill(data)
    else:
        uid = (a.shape, b.shape, wrt is a, wrt is b)
        if uid not in _bs_setup_data2:
            input_sz = wrt.size
            output_sz = np.broadcast(a.r, b.r).size
            a2 = np.arange(a.size).reshape(a.shape) if wrt is a else np.zeros(a.shape)
            b2 = np.arange(b.size).reshape(b.shape) if (wrt is b and wrt is not a) else np.zeros(b.shape)
            IS = np.arange(output_sz)
            JS = np.asarray((np.add(a2,b2)).ravel(), np.uint32)

            _bs_setup_data2[uid] = sp.csc_matrix((np.arange(IS.size), (IS, JS)), shape=(output_sz, input_sz))

        result = copy_copy(_bs_setup_data2[uid])
        if isinstance(data, np.ndarray):
            result.data = data[result.data]
        else: # assumed scalar
            result.data = np.empty(result.nnz)
            result.data.fill(data)

    if np.prod(result.shape) == 1:
        return np.array(data)
    else:
        return result




broadcast_shape_cache = {}
def broadcast_shape(a_shape, b_shape):
    global broadcast_shape_cache

    raise Exception('This function is probably a bad idea, because shape is not cached and overquerying can occur.')

    uid = (a_shape, b_shape)

    if uid not in broadcast_shape_cache:
        la = len(a_shape)
        lb = len(b_shape)
        ln = la if la > lb else lb

        ash = np.ones(ln, dtype=np.uint32)
        bsh = np.ones(ln, dtype=np.uint32)
        ash[-la:] = a_shape
        bsh[-lb:] = b_shape

        our_result = np.max(np.vstack((ash, bsh)), axis=0)

        if False:
            numpy_result = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape
            #print 'aaa' + str(our_result)
            #print 'bbb' + str(numpy_result)
            if not np.array_equal(our_result, numpy_result):
                raise Exception('numpy result not equal to our result')
            assert(np.array_equal(our_result, numpy_result))

        broadcast_shape_cache[uid] = tuple(our_result)
    return broadcast_shape_cache[uid]


def _broadcast_setup(a, b, wrt):
    if len(set((a.shape, b.shape))) == 1:
        asz = a.size
        IS = np.arange(asz)
        return IS, IS, asz, asz
    input_sz = wrt.r.size
    output_sz = np.broadcast(a.r, b.r).size
    a2 = np.arange(a.size).reshape(a.shape) if wrt is a else np.zeros(a.shape)
    b2 = np.arange(b.size).reshape(b.shape) if (wrt is b and wrt is not a) else np.zeros(b.shape)    
    IS = np.arange(output_sz)
    JS = np.asarray((np.add(a2,b2)).ravel(), np.uint32)
    return IS, JS, input_sz, output_sz


  
class add(Ch):
    dterms = 'a', 'b'
        
    def compute_r(self):
        return self.a.r + self.b.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return None

        m = 2. if self.a is self.b else 1.
        return _broadcast_matrix(self.a, self.b, wrt, m)


            
            
class subtract(Ch):
    dterms = 'a', 'b'

    def compute_r(self):
        return self.a.r - self.b.r

    def compute_dr_wrt(self, wrt):
        if (wrt is self.a) == (wrt is self.b):
            return None

        m = 1. if wrt is self.a else -1.
        return _broadcast_matrix(self.a, self.b, wrt, m)


    
    
    
class power (Ch):
    """Given vector \f$x\f$, computes \f$x^2\f$ and \f$\frac{dx^2}{x}\f$"""
    dterms = 'x', 'pow'

    def compute_r(self):
        return self.safe_power(self.x.r, self.pow.r)

    def compute_dr_wrt(self, wrt):

        if wrt is not self.x and wrt is not self.pow:
            return None
            
        x, pow = self.x.r, self.pow.r
        result = []
        if wrt is self.x:
            result.append(pow * self.safe_power(x, pow-1.))
        if wrt is self.pow:
            result.append(np.log(x) * self.safe_power(x, pow))
            
        data = reduce(lambda x, y : x + y, result).ravel()

        return _broadcast_matrix(self.x, self.pow, wrt, data)

    
    def safe_power(self, x, sigma):
        # This throws a RuntimeWarning sometimes, but then the infs are corrected below
        result = np.power(x, sigma)
        result.ravel()[np.isinf(result.ravel())] = 0
        return result



        

class A_extremum(Ch):
    """Superclass for various min and max subclasses"""
    dterms = 'a'
    terms = 'axis'
    term_order = 'a', 'axis'

    def f(self, axis):    raise NotImplementedError
    def argf(self, axis): raise NotImplementedError

    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
    
    def compute_r(self):        
        return self.f(self.a.r, axis=self.axis)
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.a:

            mn, stride = self._stride_for_axis(self.axis, self.a.r)
            JS = np.asarray(np.round(mn + stride * self.argf(self.a.r, axis=self.axis)), dtype=np.uint32).ravel()
            IS = np.arange(JS.size)
            data = np.ones(JS.size)
            
            if self.r.size * wrt.r.size == 1:
                return data.ravel()[0]
            return sp.csc_matrix((data, (IS, JS)), shape = (self.r.size, wrt.r.size))
            
    def _stride_for_axis(self,axis, mtx):
        if axis is None:
            mn = np.array([0])
            stride = np.array([1])
        else:    
            # TODO: make this less expensive. Shouldn't need to call
            # np.amin here probably
            idxs = np.arange(mtx.size).reshape(mtx.shape)
            mn = np.amin(idxs, axis=axis)
            mtx_strides = np.array(mtx.strides)
            stride = mtx_strides / np.min(mtx_strides) # go from bytes to num elements
            stride = stride[axis]
        return mn, stride


class amax(A_extremum):
    def f(self, *args, **kwargs):    return np.amax(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.argmax(*args, **kwargs)
    
max = amax    

class amin(A_extremum):
    def f(self, *args, **kwargs):    return np.amin(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.argmin(*args, **kwargs)

min = amin

class nanmin(A_extremum):
    def f(self, *args, **kwargs):    return np.nanmin(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.nanargmin(*args, **kwargs)

class nanmax(A_extremum):
    def f(self, *args, **kwargs):    return np.nanmax(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.nanargmax(*args, **kwargs)
    

class Extremum(Ch):
    dterms = 'a','b'
    
    def compute_r(self): return self.f(self.a.r, self.b.r)
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return None
                
        IS, JS, input_sz, output_sz = _broadcast_setup(self.a, self.b, wrt)
        if wrt is self.a:
            whichmax = (self.r == self.f(self.a.r, self.b.r-self.f(1,-1))).ravel()
        else:
            whichmax = (self.r == self.f(self.b.r, self.a.r-self.f(1,-1))).ravel()
        IS = IS[whichmax]
        JS = JS[whichmax]
        data = np.ones(JS.size)
        
        return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))

class maximum(Extremum):
    def f(self, a, b): return np.maximum(a, b)
    
class minimum(Extremum):
    def f(self, a, b): return np.minimum(a, b)


class multiply(Ch):
    dterms = 'a', 'b'

    def compute_r(self):
        return self.a.r * self.b.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return None
        
        a2 = self.a.r if wrt is self.b else np.ones(self.a.shape)
        b2 = self.b.r if (wrt is self.a and wrt is not self.b) else np.ones(self.b.shape)
        data = (a2 * b2).ravel()
        
        if self.a is self.b:
            data *= 2.

        return _broadcast_matrix(self.a, self.b, wrt, data)


        
                
        
class dot(Ch):
    dterms = 'a', 'b'

    def compute_r(self):
        return self.a.r.dot(self.b.r)
    
    def compute_d1(self):
        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        ar = row(self.a.r) if len(self.a.r.shape)<2 else self.a.r.reshape((-1, self.a.r.shape[-1]))
        br = col(self.b.r) if len(self.b.r.shape)<2 else self.b.r.reshape((self.b.r.shape[0], -1))

        if ar.ndim <= 2:
            return sp.kron(sp.eye(ar.shape[0], ar.shape[0]),br.T)
        else:
            raise NotImplementedError

    def compute_d2(self):
        
        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        ar = row(self.a.r) if len(self.a.r.shape)<2 else self.a.r.reshape((-1, self.a.r.shape[-1]))
        br = col(self.b.r) if len(self.b.r.shape)<2 else self.b.r.reshape((self.b.r.shape[0], -1))

        if br.ndim <= 1:
            return self.ar
        elif br.ndim <= 2:
            return sp.kron(ar, sp.eye(br.shape[1],br.shape[1]))
        else:
            raise NotImplementedError
            
    
    def compute_dr_wrt(self, wrt):

        if wrt is self.a and wrt is self.b:
            return self.compute_d1() + self.compute_d2()
        elif wrt is self.a:
            return self.compute_d1()
        elif wrt is self.b:
            return self.compute_d2()
        
class BinaryElemwiseNoDrv(Ch):
    dterms = 'x1', 'x2'
    
    def compute_r(self):
        return self._f(self.x1.r, self.x2.r)
    
    def compute_dr_wrt(self, wrt):
        return None
    
class greater(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.greater(a,b)

class greater_equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.greater_equal(a,b)

class less(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.less(a,b)

class less_equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.less_equal(a,b)
    
class equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.equal(a,b)

class not_equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.not_equal(a,b)
    
def nonzero(a):
    if hasattr(a, 'compute_r'):
        a = a.r
    return np.nonzero(a)

# Pull the code for tensordot in from numpy and reinterpret it using chumpy ops
import os
source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'np_tensordot.py')
with open(source_path, 'r') as f:
    source_lines = f.readlines()
exec(''.join(source_lines))
__all__ += ['tensordot']



def main():
    pass


if __name__ == '__main__':
    main()

