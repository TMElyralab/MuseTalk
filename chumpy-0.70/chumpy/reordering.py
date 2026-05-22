"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

from .ch import Ch
import numpy as np
from .utils import row, col
import scipy.sparse as sp
import weakref

__all__ = ['sort', 'tile', 'repeat', 'transpose', 'rollaxis', 'swapaxes', 'reshape', 'Select',
           'atleast_1d', 'atleast_2d', 'atleast_3d', 'squeeze', 'expand_dims', 'fliplr', 'flipud',
           'concatenate', 'vstack', 'hstack', 'dstack', 'ravel', 'diag', 'diagflat', 'roll', 'rot90']

# Classes deriving from "Permute" promise to only reorder/reshape
class Permute(Ch):
    pass

def ravel(a, order='C'):
    assert(order=='C')
    if isinstance (a, np.ndarray):
        self = Ch(a)

    return reshape(a=a, newshape=(-1,))

class Reorder(Permute):
    dterms = 'a',

    def on_changed(self, which):
        if not hasattr(self, 'dr_lookup'):
            self.dr_lookup = {}
            
    def compute_r(self):
        return self.reorder(self.a.r)
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.a:
            if False:
                from scipy.sparse.linalg.interface import LinearOperator
                return LinearOperator((self.size, wrt.size), lambda x : self.reorder(x.reshape(self.a.shape)).ravel())
            else:
                a = self.a
                asz = a.size
                ashape = a.shape
                key = self.unique_reorder_id()
                if key not in self.dr_lookup or key is None:
                    JS = self.reorder(np.arange(asz).reshape(ashape))
                    IS = np.arange(JS.size)
                    data = np.ones_like(IS)
                    shape = JS.shape
                    self.dr_lookup[key] = sp.csc_matrix((data, (IS, JS.ravel())), shape=(self.r.size, wrt.r.size))
                return self.dr_lookup[key]
                 
class Sort(Reorder):
    dterms = 'a'
    terms = 'axis', 'kind', 'order'
    
    def reorder(self, a): return np.sort(a, self.axis, self.kind, self.order)
    def unique_reorder_id(self): return None

def sort(a, axis=-1, kind='quicksort', order=None):
    return Sort(a=a, axis=axis, kind=kind, order=order)
    
    
class Tile(Reorder):
    dterms = 'a',
    terms = 'reps',
    term_order = 'a', 'reps'
    
    def reorder(self, a): return np.tile(a, self.reps)
    def unique_reorder_id(self): return (self.a.shape, tuple(self.reps))
    
def tile(A, reps):
    return Tile(a=A, reps=reps)
    
    
class Diag(Reorder):
    dterms = 'a',
    terms = 'k',
    
    def reorder(self, a): return np.diag(a, self.k)
    def unique_reorder_id(self): return (self.a.shape, self.k)
    
def diag(v, k=0):
    return Diag(a=v, k=k)
    
class DiagFlat(Reorder):
    dterms = 'a',
    terms = 'k',
    
    def reorder(self, a): return np.diagflat(a, self.k)
    def unique_reorder_id(self): return (self.a.shape, self.k)

def diagflat(v, k=0):
    return DiagFlat(a=v, k=k)
    
    
class Repeat(Reorder):
    dterms = 'a',
    terms = 'repeats', 'axis'
    
    def reorder(self, a): return np.repeat(a, self.repeats, self.axis)
    def unique_reorder_id(self): return (self.repeats, self.axis)

def repeat(a, repeats, axis=None):
    return Repeat(a=a, repeats=repeats, axis=axis)

class transpose(Reorder):        
    dterms = 'a'
    terms = 'axes'
    term_order = 'a', 'axes'

    def reorder(self, a):    return np.require(np.transpose(a, axes=self.axes), requirements='C')        
    def unique_reorder_id(self): return (self.a.shape, None if self.axes is None else tuple(self.axes))
    def on_changed(self, which):
        if not hasattr(self, 'axes'):
            self.axes = None
        super(self.__class__, self).on_changed(which)
    
class rollaxis(Reorder):        
    dterms = 'a'
    terms = 'axis', 'start'
    term_order = 'a', 'axis', 'start'

    def reorder(self, a):    return np.rollaxis(a, axis=self.axis, start=self.start)
    def unique_reorder_id(self): return (self.a.shape, self.axis, self.start)
    def on_changed(self, which):
        if not hasattr(self, 'start'):
            self.start = 0
        super(self.__class__, self).on_changed(which)

class swapaxes(Reorder):        
    dterms = 'a'
    terms = 'axis1', 'axis2'
    term_order = 'a', 'axis1', 'axis2'

    def reorder(self, a):    return np.swapaxes(a, axis1=self.axis1, axis2=self.axis2)
    def unique_reorder_id(self): return (self.a.shape, self.axis1, self.axis2)
    


class Roll(Reorder):
    dterms = 'a',
    terms = 'shift', 'axis'
    term_order = 'a', 'shift', 'axis'
    
    def reorder(self, a): return np.roll(a, self.shift, self.axis)
    def unique_reorder_id(self): return (self.shift, self.axis)
    
def roll(a, shift, axis=None):
    return Roll(a, shift, axis)

class Rot90(Reorder):
    dterms = 'a',
    terms = 'k',
    
    def reorder(self, a): return np.rot90(a, self.k)
    def unique_reorder_id(self): return (self.a.shape, self.k)

def rot90(m, k=1):
    return Rot90(a=m, k=k)

class Reshape(Permute):
    dterms = 'a',
    terms = 'newshape',
    term_order= 'a', 'newshape'

    def compute_r(self):
        return self.a.r.reshape(self.newshape)

    def compute_dr_wrt(self, wrt):
        if wrt is self.a:
            return sp.eye(self.a.size, self.a.size)
        #return self.a.dr_wrt(wrt)

# def reshape(a, newshape):
#     if isinstance(a, Reshape) and a.newshape == newshape:
#         return a
#     return Reshape(a=a, newshape=newshape)
def reshape(a, newshape):
    while isinstance(a, Reshape):
        a = a.a
    return Reshape(a=a, newshape=newshape)

# class And(Ch):
#     dterms = 'x1', 'x2'
#
#     def compute_r(self):
#         if True:
#             needs_work = [self.x1, self.x2]
#             done = []
#             while len(needs_work) > 0:
#                 todo = needs_work.pop()
#                 if isinstance(todo, And):
#                     needs_work += [todo.x1, todo.x2]
#                 else:
#                     done = [todo] + done
#             return np.concatenate([d.r.ravel() for d in done])
#         else:
#             return np.concatenate((self.x1.r.ravel(), self.x2.r.ravel()))
#
#     # This is only here for reverse mode to work.
#     # Most of the time, the overridden dr_wrt is callpath gets used.
#     def compute_dr_wrt(self, wrt):
#
#         if wrt is not self.x1 and wrt is not self.x2:
#             return
#
#         input_len = wrt.r.size
#         x1_len = self.x1.r.size
#         x2_len = self.x2.r.size
#
#         mtxs = []
#         if wrt is self.x1:
#             mtxs.append(sp.spdiags(np.ones(x1_len), 0, x1_len, x1_len))
#         else:
#             mtxs.append(sp.csc_matrix((x1_len, input_len)))
#
#         if wrt is self.x2:
#             mtxs.append(sp.spdiags(np.ones(x2_len), 0, x2_len, x2_len))
#         else:
#             mtxs.append(sp.csc_matrix((x2_len, input_len)))
#
#
#         if any([sp.issparse(mtx) for mtx in mtxs]):
#             result = sp.vstack(mtxs, format='csc')
#         else:
#             result = np.vstack(mtxs)
#
#         return result
#
#     def dr_wrt(self, wrt, want_stacks=False, reverse_mode=False):
#         self._call_on_changed()
#
#         input_len = wrt.r.size
#         x1_len = self.x1.r.size
#         x2_len = self.x2.r.size
#
#         mtxs = []
#         if wrt is self.x1:
#             mtxs.append(sp.spdiags(np.ones(x1_len), 0, x1_len, x1_len))
#         else:
#             if isinstance(self.x1, And):
#                 tmp_mtxs = self.x1.dr_wrt(wrt, want_stacks=True, reverse_mode=reverse_mode)
#                 for mtx in tmp_mtxs:
#                     mtxs.append(mtx)
#             else:
#                 mtxs.append(self.x1.dr_wrt(wrt, reverse_mode=reverse_mode))
#             if mtxs[-1] is None:
#                 mtxs[-1] = sp.csc_matrix((x1_len, input_len))
#
#         if wrt is self.x2:
#             mtxs.append(sp.spdiags(np.ones(x2_len), 0, x2_len, x2_len))
#         else:
#             if isinstance(self.x2, And):
#                 tmp_mtxs = self.x2.dr_wrt(wrt, want_stacks=True, reverse_mode=reverse_mode)
#                 for mtx in tmp_mtxs:
#                     mtxs.append(mtx)
#             else:
#                 mtxs.append(self.x2.dr_wrt(wrt, reverse_mode=reverse_mode))
#             if mtxs[-1] is None:
#                 mtxs[-1] = sp.csc_matrix((x2_len, input_len))
#
#         if want_stacks:
#             return mtxs
#         else:
#             if any([sp.issparse(mtx) for mtx in mtxs]):
#                 result = sp.vstack(mtxs, format='csc')
#             else:
#                 result = np.vstack(mtxs)
#
#         return result
        
class Select(Permute):
    terms = ['idxs', 'preferred_shape']
    dterms = ['a']
    term_order = 'a', 'idxs', 'preferred_shape'

    def compute_r(self):
        result = self.a.r.ravel()[self.idxs].copy()
        if hasattr(self, 'preferred_shape'):
            return result.reshape(self.preferred_shape)
        else:
            return result

    def compute_dr_wrt(self, obj):
        if obj is self.a:
            if not hasattr(self, '_dr_cached'):
                IS = np.arange(len(self.idxs))
                JS = self.idxs.ravel()
                ij = np.vstack((row(IS), row(JS)))
                data = np.ones(len(self.idxs))
                self._dr_cached = sp.csc_matrix((data, ij), shape=(len(self.idxs), np.prod(self.a.shape)))
            return self._dr_cached
        
    def on_changed(self, which):
        if hasattr(self, '_dr_cached'):
            if 'idxs' in which or self.a.r.size != self._dr_cached.shape[1]:
                del self._dr_cached

    

class AtleastNd(Ch):
    dterms = 'x'
    terms = 'ndims'
    
    def compute_r(self):
        xr = self.x.r
        if self.ndims == 1:
            target_shape = np.atleast_1d(xr).shape
        elif self.ndims == 2:
            target_shape = np.atleast_2d(xr).shape
        elif self.ndims == 3:
            target_shape = np.atleast_3d(xr).shape
        else:
            raise Exception('Need ndims to be 1, 2, or 3.')

        return xr.reshape(target_shape)
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return 1

def atleast_nd(ndims, *arys):
    arys = [AtleastNd(x=ary, ndims=ndims) for ary in arys]
    return arys if len(arys) > 1 else arys[0]

def atleast_1d(*arys):
    return atleast_nd(1, *arys)

def atleast_2d(*arys):
    return atleast_nd(2, *arys)

def atleast_3d(*arys):
    return atleast_nd(3, *arys)
    
def squeeze(a, axis=None):
    if isinstance(a, np.ndarray):
        return np.squeeze(a, axis)
    shape = np.squeeze(a.r, axis).shape
    return a.reshape(shape)
    
def expand_dims(a, axis):
    if isinstance(a, np.ndarray):
        return np.expand_dims(a, axis)
    shape = np.expand_dims(a.r, axis).shape
    return a.reshape(shape)
    
def fliplr(m):
    return m[:,::-1]
    
def flipud(m):
    return m[::-1,...]
    
class Concatenate(Ch):

    def on_changed(self, which):
        if not hasattr(self, 'dr_cached'):
            self.dr_cached = weakref.WeakKeyDictionary()
            
    @property
    def our_terms(self):
        if not hasattr(self, '_our_terms'):
            self._our_terms = [getattr(self, s) for s in self.dterms]
        return self._our_terms
    
    def __getstate__(self):
        # Have to get rid of WeakKeyDictionaries for serialization
        if hasattr(self, 'dr_cached'):
            del self.dr_cached
        return super(self.__class__, self).__getstate__()
    
    def compute_r(self):
        return np.concatenate([t.r for t in self.our_terms], axis=self.axis)
                    
    @property
    def everything(self):
        if not hasattr(self, '_everything'):
            self._everything = np.arange(self.r.size).reshape(self.r.shape)
            self._everything = np.swapaxes(self._everything, self.axis, 0)
        return self._everything
    
    def compute_dr_wrt(self, wrt):
        if not hasattr(self, 'dr_cached'):
            self.dr_cached = weakref.WeakKeyDictionary()
        if wrt in self.dr_cached and self.dr_cached[wrt] is not None:
            return self.dr_cached[wrt]
        
        if wrt not in self.our_terms:
            return
                        
        _JS = np.arange(wrt.size)
        _data = np.ones(wrt.size)
        
        IS = []
        JS = []
        data = []
        
        offset = 0
        for term in self.our_terms:
            tsz = term.shape[self.axis]
            if term is wrt:
                JS += [_JS]
                data += [_data]
                IS += [np.swapaxes(self.everything[offset:offset+tsz], self.axis, 0).ravel()]
            offset += tsz
        IS   = np.concatenate(IS).ravel()
        JS   = np.concatenate(JS).ravel()
        data = np.concatenate(data)
                
        res = sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.size))
        
        if len(list(self._parents.keys())) != 1:
            self.dr_cached[wrt] = res
        else:
            self.dr_cached[wrt] = None

        return res


def expand_concatenates(mtxs, axis=0):
    mtxs = list(mtxs)
    done = []
    while len(mtxs) > 0:
        mtx = mtxs.pop(0)
        if isinstance(mtx, Concatenate) and mtx.axis == axis:
            mtxs = [getattr(mtx, s) for s in mtx.dterms] + mtxs
        else:
            done.append(mtx)
    return done


def concatenate(mtxs, axis=0, **kwargs):

    mtxs = expand_concatenates(mtxs, axis)

    result = Concatenate(**kwargs)
    result.dterms = []
    for i, mtx in enumerate(mtxs):
        result.dterms.append('m%d' % (i,))
        setattr(result, result.dterms[-1], mtx)
    result.axis = axis
    return result
    
def hstack(mtxs, **kwargs):
    return concatenate(mtxs, axis=1, **kwargs)

def vstack(mtxs, **kwargs):
    return concatenate([atleast_2d(m) for m in mtxs], axis=0, **kwargs)

def dstack(mtxs, **kwargs):
    return concatenate([atleast_3d(m) for m in mtxs], axis=2, **kwargs)
