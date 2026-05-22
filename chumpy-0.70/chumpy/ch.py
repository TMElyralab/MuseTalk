#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


__all__ = ['Ch', 'depends_on', 'MatVecMult', 'ChHandle', 'ChLambda']

import os, sys, time
import inspect
import scipy.sparse as sp
import numpy as np
import numbers
import weakref
import copy as external_copy
from functools import wraps
from scipy.sparse.linalg.interface import LinearOperator
from .utils import row, col, timer, convert_inputs_to_sparse_if_necessary
import collections
from copy import deepcopy
from functools import reduce



# Turn this on if you want the profiler injected
DEBUG = False
# Turn this on to make optimizations very chatty for debugging
VERBOSE = False
def pif(msg):
    # print-if-verbose.
    if DEBUG or VERBOSE:
        sys.stdout.write(msg + '\n')

_props_for_dict = weakref.WeakKeyDictionary()
def _props_for(cls):
    if cls not in _props_for_dict:
        _props_for_dict[cls] = set([p[0] for p in inspect.getmembers(cls, lambda x : isinstance(x, property))])
    return _props_for_dict[cls]

_dep_props_for_dict = weakref.WeakKeyDictionary()      
def _dep_props_for(cls):
    if cls not in _dep_props_for_dict:
        _dep_props_for_dict[cls] = [p for p in inspect.getmembers(cls, lambda x : isinstance(x, property)) if hasattr(p[1].fget, 'deps')]
    return _dep_props_for_dict[cls]
    

_kw_conflict_dict = weakref.WeakKeyDictionary()
def _check_kw_conflict(cls): 
    if cls not in _kw_conflict_dict:
        _kw_conflict_dict[cls] = Ch._reserved_kw.intersection(set(cls.terms).union(set(cls.dterms)))
    if _kw_conflict_dict[cls]:
        raise Exception("In class %s, don't use reserved keywords in terms/dterms: %s" % (str(cls), str(kw_conflict),))


class Term(object):
    creation_counter = 0
    def __init__(self, default=None, desc=None, dr=True):
        self.default = default
        self.desc = desc
        self.dr = dr

        # Add a creation_counter, a la Django models, so we can preserve the order in which parameters are defined in the job.
        # http://stackoverflow.com/a/3288801/893113
        self.creation_counter = Term.creation_counter
        Term.creation_counter += 1


class Ch(object):
    terms = []
    dterms = ['x']
    __array_priority__ = 2.0
    _cached_parms = {}
    _setup_terms = {}
    _default_kwargs = {'make_dense' : False, 'make_sparse' : False}
    _status = "undefined"
    
    called_dr_wrt = False
    profiler = None

    ########################################################
    # Construction

    def __new__(cls, *args, **kwargs):

        if len(args) > 0 and type(args[0]) == type(lambda : 0):
            cls = ChLambda
        
        # Create empty instance
        result = super(Ch, cls).__new__(cls)

        cls.setup_terms()

        object.__setattr__(result, '_dirty_vars', set())
        object.__setattr__(result, '_itr', None)
        object.__setattr__(result, '_parents', weakref.WeakKeyDictionary())
        object.__setattr__(result, '_cache', {'r': None, 'drs': weakref.WeakKeyDictionary()})

        if DEBUG:
            object.__setattr__(result, '_cache_info', {})
            object.__setattr__(result, '_status', 'new')
        
        for name, default_val in list(cls._default_kwargs.items()):
            object.__setattr__(result, '_%s' % name, kwargs.get(name, default_val))
            if name in kwargs:
                del kwargs[name]
        
        # Set up storage that allows @depends_on to work
        #props = [p for p in inspect.getmembers(cls, lambda x : isinstance(x, property)) if hasattr(p[1].fget, 'deps')]
        props = _dep_props_for(cls)
        cpd = {}
        for p in props:
            func_name = p[0] #id(p[1].fget)
            deps = p[1].fget.deps
            cpd[func_name] = {'deps': deps, 'value': None, 'out_of_date': True}
        
        object.__setattr__(result, '_depends_on_deps', cpd)

        if cls != Ch:
            for idx, a in enumerate(args):
                kwargs[cls.term_order[idx]] = a
        elif len(args)>0:
            kwargs['x'] = np.asarray(args[0], np.float64)

        defs = {p.name : deepcopy(p.default) for p in cls.parm_declarations() if p.default is not None}
        defs.update(kwargs)
        result.set(**defs)
        
        return result

    @classmethod
    def parm_declarations(cls):
        if cls.__name__ not in cls._cached_parms:
            parameter_declarations = collections.OrderedDict()
            parameters = inspect.getmembers(cls, lambda x: isinstance(x, Term))
            for name, decl in sorted(parameters, key=lambda x: x[1].creation_counter):
                decl.name = name
                parameter_declarations[name] = decl
            cls._cached_parms[cls.__name__] = parameter_declarations
        return cls._cached_parms[cls.__name__]

    @classmethod
    def setup_terms(cls):
        if id(cls) in cls._setup_terms: return

        if cls == Ch:
            return

        parm_declarations = cls.parm_declarations()

        if cls.dterms is Ch.dterms:
            cls.dterms = []
        elif isinstance(cls.dterms, str):
            cls.dterms = (cls.dterms,)
        if cls.terms is Ch.terms:
            cls.terms = []
        elif isinstance(cls.terms, str):
            cls.terms = (cls.terms,)

        # Must be either new or old style
        len_oldstyle_parms = len(cls.dterms)+len(cls.terms)
        if len(parm_declarations) > 0:
            assert(len_oldstyle_parms==0)
            cls.term_order = [t.name for t in parm_declarations]
            cls.dterms = [t.name for t in parm_declarations if t.dr]
            cls.terms = [t.name for t in parm_declarations if not t.dr]
        else:
            if not hasattr(cls, 'term_order'):
                cls.term_order = list(cls.terms) + list(cls.dterms)

        _check_kw_conflict(cls)
        cls._setup_terms[id(cls)] = True


    ########################################################
    # Identifiers

    @property
    def short_name(self):
        return self.label if hasattr(self, 'label') else self.__class__.__name__

    @property
    def sid(self):
        """Semantic id."""
        pnames = list(self.terms)+list(self.dterms)
        pnames.sort()
        return (self.__class__, tuple([(k, id(self.__dict__[k])) for k in pnames if k in self.__dict__]))

    
    def reshape(self, *args):
        return reshape(a=self, newshape=args if len(args)>1 else args[0])
    
    def ravel(self):
        return reshape(a=self, newshape=(-1))
    
    def __hash__(self):
        return id(self)
        
    @property
    def ndim(self):
        return self.r.ndim
        
    @property
    def flat(self):
        return self.r.flat
        
    @property
    def dtype(self):
        return self.r.dtype
        
    @property
    def itemsize(self):
        return self.r.itemsize
            

    ########################################################
    # Redundancy removal

    def remove_redundancy(self, cache=None, iterate=True):

        if cache == None:
            cache = {}
            _ = self.r # may result in the creation of extra dterms that we can cull
            
        replacement_occurred = False
        for propname in list(self.dterms):
            prop = self.__dict__[propname]

            if not hasattr(prop, 'dterms'):
                continue
            sid = prop.sid
            if sid not in cache:
                cache[sid] = prop
            elif self.__dict__[propname] is not cache[sid]:
                self.__dict__[propname] = cache[sid]
                replacement_occurred = True
            if prop.remove_redundancy(cache, iterate=False):
                replacement_occurred = True
                
        if not replacement_occurred:
            return False
        else:
            if iterate:
                self.remove_redundancy(cache, iterate=True)
                return False
            else:
                return True
            
                
                
    def print_labeled_residuals(self, print_newline=True, num_decimals=2, where_to_print=None):
        
        if where_to_print is None:
            where_to_print = sys.stderr
        if hasattr(self, 'label'):
            where_to_print.write(('%s: %.' + str(num_decimals) + 'e | ') % (self.label, np.sum(self.r**2)))
        for dterm in self.dterms:
            dt = getattr(self, dterm)            
            if hasattr(dt, 'dterms'):
                dt.print_labeled_residuals(print_newline=False, where_to_print=where_to_print)            
        if print_newline:
            where_to_print.write(('%.' + str(num_decimals) + 'e\n') % (np.sum(self.r**2),))
        
    

    ########################################################
    # Default methods, for when Ch is not subclassed

    def compute_r(self):
        """Default method for objects that just contain a number or ndarray"""
        return self.x
        
    def compute_dr_wrt(self,wrt):
        """Default method for objects that just contain a number or ndarray"""
        if wrt is self: # special base case  
            return sp.eye(self.x.size, self.x.size)
            #return np.array([[1]])
        return None
        
    
    def _compute_dr_wrt_sliced(self, wrt):
        self._call_on_changed()
        
        # if wrt is self:
        #     return np.array([[1]])

        result = self.compute_dr_wrt(wrt)
        if result is not None:
            return result

        # What allows slicing.
        if True:
            inner = wrt
            while issubclass(inner.__class__, Permute):
                inner = inner.a
                if inner is self: 
                    return None
                result = self.compute_dr_wrt(inner)

                if result is not None:
                    break
        
            if result is None:
                return None
        
            wrt._call_on_changed()

            jac = wrt.compute_dr_wrt(inner).T

            return self._superdot(result, jac)


    @property
    def shape(self):
        return self.r.shape
    
    @property
    def size(self):
        #return self.r.size
        return np.prod(self.shape) # may be cheaper since it doesn't always mean grabbing "r"
    
    def __len__(self):
        return len(self.r)
        
    def minimize(self, *args, **kwargs):
        from . import optimization        
        return optimization.minimize(self, *args, **kwargs)
        
    def __array__(self, *args):
        return self.r
    
    ########################################################
    # State management

    def add_dterm(self, dterm_name, dterm):
        self.dterms = list(set(list(self.dterms) + [dterm_name]))
        setattr(self, dterm_name, dterm)
    
    def copy(self):
        return copy(self)
    
    def __getstate__(self):
        # Have to get rid of WeakKeyDictionaries for serialization
        result = external_copy.copy(self.__dict__)
        del result['_parents']
        del result['_cache']
        return result

    def __setstate__(self, d):
        # Restore unpickleable WeakKeyDictionaries
        d['_parents'] = weakref.WeakKeyDictionary()
        d['_cache'] = {'r': None, 'drs': weakref.WeakKeyDictionary()}
        object.__setattr__(self, '__dict__', d)

        # This restores our unpickleable "_parents" attribute
        for k in set(self.dterms).intersection(set(self.__dict__.keys())):
            setattr(self, k, self.__dict__[k])
       
    def __setattr__(self, name, value, itr=None):
        #print 'SETTING %s' % (name,)

        # Faster path for basic Ch objects. Not necessary for functionality,
        # but improves performance by a small amount.
        if type(self) == Ch:
            if name == 'x':
                self._dirty_vars.add(name)
                self.clear_cache(itr)
            #else:
            #    import warnings
            #    warnings.warn('Trying to set attribute %s on a basic Ch object? Might be a mistake.' % (name,))

            object.__setattr__(self, name, value)
            return

        name_in_dterms = name in self.dterms
        name_in_terms = name in self.terms
        name_in_props = name in _props_for(self.__class__)# [p[0] for p in inspect.getmembers(self.__class__, lambda x : isinstance(x, property))]
        
        if name_in_dterms and not name_in_props and type(self) != Ch:
            if not hasattr(value, 'dterms'):
                value = Ch(value)                
        
            # Make ourselves not the parent of the old value
            if hasattr(self, name):
                term = getattr(self, name)
                if self in term._parents:
                    term._parents[self]['varnames'].remove(name)
                    if len(term._parents[self]['varnames']) == 0:
                        del term._parents[self]
                    
            # Make ourselves parents of the new value
            if self not in value._parents:
                value._parents[self] = {'varnames': set([name])}
            else:
                value._parents[self]['varnames'].add(name)

        if name_in_dterms or name_in_terms:            
            self._dirty_vars.add(name)
            self._invalidate_cacheprop_names([name])

            # If one of our terms has changed, it has the capacity to have
            # changed our result and all our derivatives wrt everything
            self.clear_cache(itr)
            
        object.__setattr__(self, name, value)
          
    def _invalidate_cacheprop_names(self, names):
        nameset = set(names)
        for func_name, v in list(self._depends_on_deps.items()):
            if len(nameset.intersection(v['deps'])) > 0:
                v['out_of_date'] = True
        
        
    def clear_cache(self, itr=None):
        todo = [self]
        done = set([])
        nodes_visited = 0
        while len(todo) > 0:
            nodes_visited += 1
            next = todo.pop()
            if itr is not None and itr==next._itr:
                continue
            if id(next) not in done:
                next._cache['r'] = None
                next._cache['drs'].clear()
                next._itr = itr

                for parent, parent_dict in list(next._parents.items()):
                    object.__setattr__(parent, '_dirty_vars', parent._dirty_vars.union(parent_dict['varnames']))
                    parent._invalidate_cacheprop_names(parent_dict['varnames']) 
                    todo.append(parent)
                done.add(id(next))
        return nodes_visited


    def clear_cache_wrt(self, wrt, itr=None):
        if wrt in self._cache['drs']:
            self._cache['drs'][wrt] = None

        if hasattr(self, 'dr_cached') and wrt in self.dr_cached:
            self.dr_cached[wrt] = None

        if itr is None or itr != self._itr:
            for parent, parent_dict in list(self._parents.items()):
                if wrt in parent._cache['drs'] or (hasattr(parent, 'dr_cached') and wrt in parent.dr_cached):
                    parent.clear_cache_wrt(wrt=wrt, itr=itr)
                object.__setattr__(parent, '_dirty_vars', parent._dirty_vars.union(parent_dict['varnames']))
                parent._invalidate_cacheprop_names(parent_dict['varnames'])

        object.__setattr__(self, '_itr', itr)

    def replace(self, old, new):
        if (hasattr(old, 'dterms') != hasattr(new, 'dterms')):
            raise Exception('Either "old" and "new" must both be "Ch", or they must both be neither.')
        
        for term_name in [t for t in list(self.dterms)+list(self.terms) if hasattr(self, t)]:
            term = getattr(self, term_name)
            if term is old:
                setattr(self, term_name, new)
            elif hasattr(term, 'dterms'):
                term.replace(old, new)
        return new
  
    
    def set(self, **kwargs):
        # Some dterms may be aliases via @property.
        # We want to set those last, in case they access non-property members
        #props = [p[0] for p in inspect.getmembers(self.__class__, lambda x : isinstance(x, property))]
        props = _props_for(self.__class__)
        kwarg_keys = set(kwargs.keys())
        kwsecond = kwarg_keys.intersection(props)
        kwfirst = kwarg_keys.difference(kwsecond)
        kwall = list(kwfirst) + list(kwsecond)

        # The complexity here comes because we wish to
        # avoid clearing cache redundantly
        if len(kwall) > 0:
            for k in kwall[:-1]:
                self.__setattr__(k, kwargs[k], 9999)
            self.__setattr__(kwall[-1], kwargs[kwall[-1]], None)
            

    def is_dr_wrt(self, wrt):
        if type(self) == Ch:
            return wrt is self
        dterms_we_have = [getattr(self, dterm) for dterm in self.dterms if hasattr(self, dterm)]
        return wrt in dterms_we_have or any([d.is_dr_wrt(wrt) for d in dterms_we_have])
    

    def is_ch_baseclass(self):
        return self.__class__ is Ch
        

    ########################################################
    # Getters for our outputs

    def __getitem__(self, key):        
        shape = self.shape
        tmp = np.arange(np.prod(shape)).reshape(shape).__getitem__(key)
        idxs = tmp.ravel()
        newshape = tmp.shape
        return Select(a=self, idxs=idxs, preferred_shape=newshape)
        
    def __setitem__(self, key, value, itr=None): 

        if hasattr(value, 'dterms'):
            raise Exception("Can't assign a Ch objects as a subset of another.")
        if type(self) == Ch:# self.is_ch_baseclass():
            data = np.atleast_1d(self.x)
            data.__setitem__(key, value)
            self.__setattr__('x', data, itr=itr)
            return
        # elif False: # Interesting but flawed idea
            # parents = [self.__dict__[k] for k in self.dterms]
            # kids = []
            # while len(parents)>0:
            #     p = parents.pop()
            #     if p.is_ch_baseclass():
            #         kids.append(p)
            #     else:
            #         parents += [p.__dict__[k] for k in p.dterms]
            # from ch.optimization import minimize_dogleg
            # minimize_dogleg(obj=self.__getitem__(key) - value, free_variables=kids, show_residuals=False)            
        else:
            inner = self
            while not inner.is_ch_baseclass():
                if issubclass(inner.__class__, Permute):
                    inner = inner.a
                else: 
                    raise Exception("Can't set array that is function of arrays.")

            self = self[key]
            dr = self.dr_wrt(inner)
            dr_rev = dr.T
            #dr_rev = np.linalg.pinv(dr)
            inner_shape = inner.shape

            t1 = self._superdot(dr_rev, np.asarray(value).ravel())
            t2 = self._superdot(dr_rev, self._superdot(dr, inner.x.ravel()))
            if sp.issparse(t1): t1 = np.array(t1.todense())
            if sp.issparse(t2): t2 = np.array(t2.todense())

            inner.x = inner.x + t1.reshape(inner_shape) - t2.reshape(inner_shape)
            #inner.x = inner.x + self._superdot(dr_rev, value.ravel()).reshape(inner_shape) - self._superdot(dr_rev, self._superdot(dr, inner.x.ravel())).reshape(inner_shape)


    def __str__(self):
        return str(self.r)

    def __repr__(self):
        return object.__repr__(self) + '\n' + str(self.r)

    def __float__(self):
        return self.r.__float__()

    def __int__(self):
        return self.r.__int__()

    def on_changed(self, terms):
        pass
        
    @property
    def T(self):
        return transpose(self)
        
    def transpose(self, *axes):
        return transpose(self, *axes)
        
    def squeeze(self, axis=None):
        return squeeze(self, axis)
        
    def mean(self, axis=None):
        return mean(self, axis=axis)

    def sum(self, axis=None):
        return sum(self, axis=axis)

    def _call_on_changed(self):

        if hasattr(self, 'is_valid'):
            validity, msg = self.is_valid()
            assert validity, msg
        if hasattr(self, '_status'):
            self._status = 'new'

        if len(self._dirty_vars) > 0:
            self.on_changed(self._dirty_vars)
            object.__setattr__(self, '_dirty_vars', set())

    @property
    def r(self):
        self._call_on_changed()
        if self._cache['r'] is None:
            self._cache['r'] = np.asarray(np.atleast_1d(self.compute_r()), dtype=np.float64, order='C')
            self._cache['rview'] = self._cache['r'].view()
            self._cache['rview'].flags.writeable = False
        
        return self._cache['rview']

    def _superdot(self, lhs, rhs, profiler=None):

        try:
            if lhs is None:
                return None
            if rhs is None:
                return None
            
            if isinstance(lhs, np.ndarray) and lhs.size==1:
                lhs = lhs.ravel()[0]
                
            if isinstance(rhs, np.ndarray) and rhs.size==1:
                rhs = rhs.ravel()[0]
    
            if isinstance(lhs, numbers.Number) or isinstance(rhs, numbers.Number):
                return lhs * rhs

            if isinstance(rhs, LinearOperator):
                return LinearOperator((lhs.shape[0], rhs.shape[1]), lambda x : lhs.dot(rhs.dot(x)))

            if isinstance(lhs, LinearOperator):                
                if sp.issparse(rhs):
                    return LinearOperator((lhs.shape[0], rhs.shape[1]), lambda x : lhs.dot(rhs.dot(x)))
                else:
                    # TODO: ?????????????
                    # return lhs.matmat(rhs)
                    return lhs.dot(rhs)
            
            # TODO: Figure out how/whether to do this.
            tm_maybe_sparse = timer()
            lhs, rhs = convert_inputs_to_sparse_if_necessary(lhs, rhs)
            if tm_maybe_sparse() > 0.1:
                pif('convert_inputs_to_sparse_if_necessary in {}sec'.format(tm_maybe_sparse()))

            if not sp.issparse(lhs) and sp.issparse(rhs):
                return rhs.T.dot(lhs.T).T
            return lhs.dot(rhs)
        except Exception as e:
            import sys, traceback
            traceback.print_exc(file=sys.stdout)
            if DEBUG:
                import pdb; pdb.post_mortem()
            else:
                raise
            
    def lmult_wrt(self, lhs, wrt):
        if lhs is None:
            return None
        
        self._call_on_changed()

        drs = []        

        direct_dr = self._compute_dr_wrt_sliced(wrt)

        if direct_dr != None:
            drs.append(self._superdot(lhs, direct_dr))

        for k in set(self.dterms):
            p = self.__dict__[k]

            if hasattr(p, 'dterms') and p is not wrt and p.is_dr_wrt(wrt):
                if not isinstance(p, Ch):
                    print('BROKEN!')
                    raise Exception('Broken Should be Ch object')

                indirect_dr = p.lmult_wrt(self._superdot(lhs, self._compute_dr_wrt_sliced(p)), wrt)
                if indirect_dr is not None:
                    drs.append(indirect_dr)

        if len(drs)==0:
            result = None

        elif len(drs)==1:
            result = drs[0]

        else:
            result = reduce(lambda x, y: x+y, drs)

        return result
        
        
    def compute_lop(self, wrt, lhs):
        dr = self._compute_dr_wrt_sliced(wrt)
        if dr is None: return None
        return self._superdot(lhs, dr) if not isinstance(lhs, LinearOperator) else lhs.matmat(dr)
        
        
    def lop(self, wrt, lhs):
        self._call_on_changed()
        
        drs = []
        direct_dr = self.compute_lop(wrt, lhs)
        if direct_dr is not None:
            drs.append(direct_dr)

        for k in set(self.dterms):
            p = getattr(self, k) # self.__dict__[k]
            if hasattr(p, 'dterms') and p is not wrt: # and p.is_dr_wrt(wrt):
                lhs_for_child = self.compute_lop(p, lhs)
                if lhs_for_child is not None: # Can be None with ChLambda, _result etc
                    indirect_dr = p.lop(wrt, lhs_for_child)
                    if indirect_dr is not None:
                        drs.append(indirect_dr)

        for k in range(len(drs)):
            if sp.issparse(drs[k]):
                drs[k] = drs[k].todense()

        if len(drs)==0:
            result = None

        elif len(drs)==1:
            result = drs[0]

        else:
            result = reduce(lambda x, y: x+y, drs)

            
        return result
        
    def compute_rop(self, wrt, rhs):
        dr = self._compute_dr_wrt_sliced(wrt)
        if dr is None: return None
        
        return self._superdot(dr, rhs)

    def dr_wrt(self, wrt, reverse_mode=False, profiler=None):
        tm_dr_wrt = timer()
        self.called_dr_wrt = True
        self._call_on_changed()

        drs = []

        if wrt in self._cache['drs']:
            if DEBUG:
                if wrt not in self._cache_info:
                    self._cache_info[wrt] = 0
                self._cache_info[wrt] +=1
                self._status = 'cached'
            return self._cache['drs'][wrt]

        direct_dr = self._compute_dr_wrt_sliced(wrt)

        if direct_dr is not None:
            drs.append(direct_dr)    

        if DEBUG:
            self._status = 'pending'
        
        propnames = set(_props_for(self.__class__))
        for k in set(self.dterms).intersection(propnames.union(set(self.__dict__.keys()))):

            p = getattr(self, k)

            if hasattr(p, 'dterms') and p is not wrt:

                indirect_dr = None

                if reverse_mode:
                    lhs = self._compute_dr_wrt_sliced(p)
                    if isinstance(lhs, LinearOperator):
                        tm_dr_wrt.pause()
                        dr2 = p.dr_wrt(wrt)
                        tm_dr_wrt.resume()
                        indirect_dr = lhs.matmat(dr2) if dr2 != None else None
                    else:
                        indirect_dr = p.lmult_wrt(lhs, wrt)
                else: # forward mode
                    tm_dr_wrt.pause()
                    dr2 = p.dr_wrt(wrt, profiler=profiler)
                    tm_dr_wrt.resume()
                    if dr2 is not None:
                        indirect_dr = self.compute_rop(p, rhs=dr2)

                if indirect_dr is not None:
                    drs.append(indirect_dr)

        if len(drs)==0:
            result = None
        elif len(drs)==1:
            result = drs[0]
        else:
            # TODO: ????????
            # result = np.sum(x for x in drs)
            if not np.any([isinstance(a, LinearOperator) for a in drs]):
                result = reduce(lambda x, y: x+y, drs)
            else:
                result = LinearOperator(drs[0].shape, lambda x : reduce(lambda a, b: a.dot(x)+b.dot(x),drs))

        # TODO: figure out how/whether to do this.
        if result is not None and not sp.issparse(result):
            tm_nonzero = timer()
            nonzero = np.count_nonzero(result)
            if tm_nonzero() > 0.1:
                pif('count_nonzero in {}sec'.format(tm_nonzero()))
            if nonzero == 0 or hasattr(result, 'size') and result.size / float(nonzero) >= 10.0:
                tm_convert_to_sparse = timer()
                result = sp.csc_matrix(result)
                import gc
                gc.collect()
                pif('converting result to sparse in {}sec'.format(tm_convert_to_sparse()))
                        
        if (result is not None) and (not sp.issparse(result)) and (not isinstance(result, LinearOperator)):
            result = np.atleast_2d(result)
        
        # When the number of parents is one, it indicates that
        # caching this is probably not useful because not 
        # more than one parent will likely ask for this same
        # thing again in the same iteration of an optimization.
        #
        # When the number of parents is zero, this is the top
        # level object and should be cached; when it's > 1
        # cache the combinations of the children.
        #
        # If we *always* filled in the cache, it would require 
        # more memory but would occasionally save a little cpu,
        # on average.
        if len(list(self._parents.keys())) != 1:
            self._cache['drs'][wrt] = result

        if DEBUG:
            self._status = 'done'
        
        if getattr(self, '_make_dense', False) and sp.issparse(result):
            result = result.todense()
        if getattr(self, '_make_sparse', False) and not sp.issparse(result):
            result = sp.csc_matrix(result)
        
        if tm_dr_wrt() > 0.1:
            pif('dx of {} wrt {} in {}sec, sparse: {}'.format(self.short_name, wrt.short_name, tm_dr_wrt(), sp.issparse(result)))

        return result


    def __call__(self, **kwargs):
        self.set(**kwargs)
        return self.r


    ########################################################
    # Visualization

    @property
    def reset_flag(self):
        """
        Used as fn in loop_children_do
        """
        return lambda x: setattr(x, 'called_dr_wrt', False)

    def loop_children_do(self, fn):
        fn(self)
        for dterm in self.dterms:
            if hasattr(self, dterm):
                dtval = getattr(self, dterm)
                if hasattr(dtval, 'dterms') or hasattr(dtval, 'terms'):
                    if hasattr(dtval, 'loop_children_do'):
                        dtval.loop_children_do(fn)


    def show_tree_cache(self, label, current_node=None):
        '''
        Show tree and cache info with color represent _status
        Optionally accpet current_node arg to highlight the current node we are in
        '''
        import os
        import tempfile
        import subprocess

        assert DEBUG, "Please use dr tree visualization functions in debug mode"

        cache_path = os.path.abspath('profiles')
        def string_for(self, my_name):

            color_mapping = {'new' : 'grey', 'pending':'red', 'cached':'yellow', 'done': 'green'}
            if hasattr(self, 'label'):
                my_name = self.label
            my_name = '%s (%s)' % (my_name, str(self.__class__.__name__))
            result = []
            if not hasattr(self, 'dterms'):
                return result
            for dterm in self.dterms:
                if hasattr(self, dterm):
                    dtval = getattr(self, dterm)
                    if hasattr(dtval, 'dterms') or hasattr(dtval, 'terms'):
                        child_label = getattr(dtval, 'label') if hasattr(dtval, 'label') else dterm
                        child_label = '%s (%s)' % (child_label, str(dtval.__class__.__name__))
                        src = 'aaa%d' % (id(self))
                        dst = 'aaa%d' % (id(dtval))

                        s = ''
                        color = color_mapping[dtval._status] if hasattr(dtval, '_status') else 'grey'
                        if dtval == current_node:
                            color = 'blue'
                        if isinstance(dtval, Concatenate) and len(dtval.dr_cached) > 0:
                            s = 'dr_cached\n'
                            for k, v in dtval.dr_cached.items():
                                if v is not None:
                                    issparse = sp.issparse(v)
                                    size = v.size 
                                    if issparse:
                                        size = v.shape[0] * v.shape[1]
                                        nonzero = len(v.data)
                                    else:
                                        nonzero = np.count_nonzero(v)  
                                    s += '\nsparse: %s\nsize: %d\nnonzero: %d\n' % (issparse, size, nonzero)
                            # if dtval.called_dr_wrt:
                            # #   dtval.called_dr_wrt = False
                            #     color = 'brown3'
                            # else:
                            #     color = 'azure1'
                        elif len(dtval._cache['drs']) > 0:
                            s = '_cache\n'
                            
                            for k, v in dtval._cache['drs'].items():
                                if v is not None:
                                    issparse = sp.issparse(v)
                                    size = v.size
                                    if issparse:
                                        size = v.shape[0] * v.shape[1]
                                        nonzero = len(v.data)
                                    else:
                                        nonzero = np.count_nonzero(v)     

                                    s += '\nsparse: %s\nsize: %d\nnonzero: %d\n' % (issparse, size, nonzero)
                                    if hasattr(dtval, '_cache_info'):
                                        s += '\ncache hit:%s\n' % dtval._cache_info[k]
                            # if hasattr(dtval,'called_dr_wrt') and dtval.called_dr_wrt:
                            # #   dtval.called_dr_wrt = False
                            #     color = 'brown3'
                            # else:
                            #     color = 'azure1'
                        result += ['%s -> %s;' % (src, dst)]
                        # Do not overwrite src
                        #result += ['%s [label="%s"];' % (src, my_name)]
                        result += ['%s [label="%s\n%s\n", color=%s, style=filled];' %
                                   (dst, child_label, s, color)]
                        result += string_for(getattr(self, dterm), dterm)
            return result
            

        dot_file_contents = 'digraph G {\n%s\n}' % '\n'.join(list(set(string_for(self, 'root'))))
        dot_file_name = os.path.join(cache_path, label)
        png_file_name = os.path.join(cache_path, label+'.png')
        with open(dot_file_name, 'w') as dot_file:
            with open(png_file_name, 'w') as png_file:
                dot_file.write(dot_file_contents)
                dot_file.flush()

                png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                subprocess.call(['dot', '-Tpng', '-o', png_file.name, dot_file.name])

        import webbrowser
        webbrowser.open('file://' + png_file.name)
        
        self.loop_children_do(self.reset_flag)

    def show_tree_wrt(self, wrt):
        import tempfile
        import subprocess

        assert DEBUG, "Please use dr tree visualization functions in debug mode"

        def string_for(self, my_name, wrt):
            if hasattr(self, 'label'):
                my_name = self.label
            my_name = '%s (%s)' % (my_name, str(self.__class__.__name__))
            result = []
            if not hasattr(self, 'dterms'):
                return result
            for dterm in self.dterms:
                if hasattr(self, dterm):
                    dtval = getattr(self, dterm)
                    if hasattr(dtval, 'dterms') or hasattr(dtval, 'terms'):
                        child_label = getattr(dtval, 'label') if hasattr(dtval, 'label') else dterm
                        child_label = '%s (%s)' % (child_label, str(dtval.__class__.__name__))
                        src = 'aaa%d' % (id(self))
                        dst = 'aaa%d' % (id(dtval))
                        result += ['%s -> %s;' % (src, dst)]
                        result += ['%s [label="%s"];' % (src, my_name)]
                        if wrt in dtval._cache['drs'] and dtval._cache['drs'][wrt] is not None:
                            issparse = sp.issparse(dtval._cache['drs'][wrt])
                            size = dtval._cache['drs'][wrt].size 
                            nonzero = np.count_nonzero(dtval._cache['drs'][wrt])
                            result += ['%s [label="%s\n is_sparse: %s\nsize: %d\nnonzero: %d"];' %
                                       (dst, child_label, issparse, size,
                                        nonzero)]
                        else:
                            result += ['%s [label="%s"];' % (dst, child_label)]
                        result += string_for(getattr(self, dterm), dterm, wrt)
            return result
            
        
        dot_file_contents = 'digraph G {\n%s\n}' % '\n'.join(list(set(string_for(self, 'root', wrt))))
        dot_file = tempfile.NamedTemporaryFile()
        dot_file.write(dot_file_contents)
        dot_file.flush()
        png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        subprocess.call(['dot', '-Tpng', '-o', png_file.name, dot_file.name])
        import webbrowser
        webbrowser.open('file://' + png_file.name)
    
    def show_tree(self, cachelim=np.inf):
        """Cachelim is in Mb. For any cached jacobians above cachelim, they are also added to the graph. """
        import tempfile
        import subprocess

        assert DEBUG, "Please use dr tree visualization functions in debug mode"

        def string_for(self, my_name):
            if hasattr(self, 'label'):
                my_name = self.label
            my_name = '%s (%s)' % (my_name, str(self.__class__.__name__))
            result = []
            if not hasattr(self, 'dterms'):
                return result
            for dterm in self.dterms:
                if hasattr(self, dterm):
                    dtval = getattr(self, dterm)
                    if hasattr(dtval, 'dterms') or hasattr(dtval, 'terms'):
                        child_label = getattr(dtval, 'label') if hasattr(dtval, 'label') else dterm
                        child_label = '%s (%s)' % (child_label, str(dtval.__class__.__name__))
                        src = 'aaa%d' % (id(self))
                        dst = 'aaa%d' % (id(dtval))
                        result += ['%s -> %s;' % (src, dst)]
                        result += ['%s [label="%s"];' % (src, my_name)]
                        result += ['%s [label="%s"];' % (dst, child_label)]
                        result += string_for(getattr(self, dterm), dterm)

            if cachelim != np.inf and hasattr(self, '_cache') and 'drs' in self._cache:
                from six.moves import cPickle as pickle
                for dtval, jac in list(self._cache['drs'].items()):
                    # child_label = getattr(dtval, 'label') if hasattr(dtval, 'label') else dterm
                    # child_label = '%s (%s)' % (child_label, str(dtval.__class__.__name__))
                    src = 'aaa%d' % (id(self))
                    dst = 'aaa%d' % (id(dtval))
                    try:                    
                        sz = sys.getsizeof(pickle.dumps(jac, -1))
                    except: # some are functions
                        sz = 0
                    # colorattr = "#%02x%02x%02x" % (szpct*255, 0, (1-szpct)*255)
                    # print colorattr
                    if sz > (cachelim * 1024 * 1024):
                        result += ['%s -> %s [style=dotted,color="<<<%d>>>"];' % (src, dst, sz)]
                    #
                    # result += ['%s -> %s [style=dotted];' % (src, dst)]
                    # result += ['%s [label="%s"];' % (src, my_name)]
                    # result += ['%s [label="%s"];' % (dst, child_label)]
                    # result += string_for(getattr(self, dterm), dterm)
                    
            return result

        dot_file_contents = 'digraph G {\n%s\n}' % '\n'.join(list(set(string_for(self, 'root'))))
        if cachelim != np.inf:
            import re
            strs = re.findall(r'<<<(\d+)>>>', dot_file_contents, re.DOTALL)
            if len(strs) > 0:
                the_max = np.max(np.array([int(d) for d in strs]))
                for s in strs:
                    szpct = float(s)/the_max
                    sz = float(s)
                    unit = 'b'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'K'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'M'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'G'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'T'
                        
                    dot_file_contents = re.sub('<<<%s>>>' % s, '#%02x%02x%02x",label="%d%s' % (szpct*255, 0, (1-szpct)*255, sz, unit), dot_file_contents)

        dot_file = tempfile.NamedTemporaryFile()
        dot_file.write(dot_file_contents)
        dot_file.flush()
        png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        subprocess.call(['dot', '-Tpng', '-o', png_file.name, dot_file.name])
        import webbrowser
        webbrowser.open('file://' + png_file.name)


    def tree_iterator(self, visited=None, path=None):
        '''
        Generator function that traverse the dr tree start from this node (self).
        '''
        if visited is None:
            visited = set()

        if self not in visited:
            if path and isinstance(path, list):
                path.append(self)

            visited.add(self)
            yield self

            if not hasattr(self, 'dterms'):
                yield

            for dterm in self.dterms:
                if hasattr(self, dterm):
                    child = getattr(self, dterm)
                    if hasattr(child, 'dterms') or hasattr(child, 'terms'):
                        for node in child.tree_iterator(visited):
                            yield node

    def floor(self):
        return floor(self)
    
    def ceil(self):
        return ceil(self)

    def dot(self, other):
        return dot(self, other)

    def cumsum(self, axis=None):
        return cumsum(a=self, axis=axis)
        
    def min(self, axis=None):
        return amin(a=self, axis=axis)
    
    def max(self, axis=None):
        return amax(a=self, axis=axis)

    ########################################################
    # Operator overloads        

    def __pos__(self): return self    
    def __neg__(self): return negative(self)

    def __add__ (self, other): return add(a=self, b=other)
    def __radd__(self, other): return add(a=other, b=self)

    def __sub__ (self, other): return subtract(a=self, b=other)
    def __rsub__(self, other): return subtract(a=other, b=self)
        
    def __mul__ (self, other): return multiply(a=self, b=other)
    def __rmul__(self, other): return multiply(a=other, b=self)
        
    def __div__ (self, other): return divide(x1=self, x2=other)
    def __truediv__ (self, other): return divide(x1=self, x2=other)
    def __rdiv__(self, other): return divide(x1=other, x2=self)

    def __pow__ (self, other): return power(x=self, pow=other)
    def __rpow__(self, other): return power(x=other, pow=self)
        
    def __rand__(self, other): return self.__and__(other)

    def __abs__ (self): return abs(self)
    
    def __gt__(self, other): return greater(self, other)
    def __ge__(self, other): return greater_equal(self, other)
        
    def __lt__(self, other): return less(self, other)
    def __le__(self, other): return less_equal(self, other)
    
    def __ne__(self, other): return not_equal(self, other)
    
    # not added yet because of weak key dict conflicts
    #def __eq__(self, other): return equal(self, other)


Ch._reserved_kw = set(Ch.__dict__.keys())
        

class MatVecMult(Ch):
    terms = 'mtx'
    dterms = 'vec'
    def compute_r(self):
        result = self.mtx.dot(col(self.vec.r.ravel())).ravel()
        if len(self.vec.r.shape) > 1 and self.vec.r.shape[1] > 1:
            result = result.reshape((-1,self.vec.r.shape[1]))
        return result

    def compute_dr_wrt(self, wrt):
        if wrt is self.vec:
            return sp.csc_matrix(self.mtx)
        
    
#def depends_on(*dependencies):
#    def _depends_on(func):
#        @wraps(func)
#        def with_caching(self, *args, **kwargs):
#            return func(self, *args, **kwargs)
#        return property(with_caching)
#    return _depends_on
    
        
def depends_on(*dependencies):
    deps = set()
    for dep in dependencies:
        if isinstance(dep, str):
            deps.add(dep)
        else:
            [deps.add(d) for d in dep]
    
    def _depends_on(func):
        want_out = 'out' in inspect.getargspec(func).args
        
        @wraps(func)
        def with_caching(self, *args, **kwargs):
            func_name = func.__name__
            sdf = self._depends_on_deps[func_name]
            if sdf['out_of_date'] == True:
                #tm = time.time()
                if want_out: 
                    kwargs['out'] = sdf['value']
                sdf['value'] = func(self, *args, **kwargs)
                sdf['out_of_date'] = False
                #print 'recomputed %s in %.2e' % (func_name, time.time() - tm)
            return sdf['value']
        with_caching.deps = deps # set(dependencies)
        result = property(with_caching)
        return result
    return _depends_on  



class ChHandle(Ch):
    dterms = ('x',)
    
    def compute_r(self):
        assert(self.x is not self)
        return self.x.r
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return 1
    

class ChLambda(Ch):
    terms = ['lmb', 'initial_args']
    dterms = []
    term_order = ['lmb', 'initial_args']

    def on_changed(self, which):
        for argname in set(which).intersection(set(self.args.keys())):
            self.args[argname].x = getattr(self, argname)
    
    def __init__(self, lmb, initial_args=None):
        args = {argname: ChHandle(x=Ch(idx)) for idx, argname in enumerate(inspect.getargspec(lmb)[0])}
        if initial_args is not None:
            for initial_arg in initial_args:
                if initial_arg in args:
                    args[initial_arg].x = initial_args[initial_arg]        
        result = lmb(**args)
        for argname, arg in list(args.items()):
            if result.is_dr_wrt(arg.x):
                self.add_dterm(argname, arg.x)
            else:
                self.terms.append(argname)
                setattr(self, argname, arg.x)
        self.args = args
        self.add_dterm('_result', result)
        
    def __getstate__(self):
        # Have to get rid of lambda for serialization
        if hasattr(self, 'lmb'):
            self.lmb = None
        return super(self.__class__, self).__getstate__()
        
        
    def compute_r(self):
        return self._result.r
    
    def compute_dr_wrt(self, wrt):
        if wrt is self._result:
            return 1

# ChGroup is similar to ChLambda in that it's designed to expose the "internal"
# inputs of result but, unlike ChLambda, result is kept internal and called when
# compute_r and compute_dr_wrt is called to compute the relevant Jacobians.
# This provides a way of effectively applying the chain rule in a different order.
class ChGroup(Ch):
    terms = ['result', 'args']
    dterms = []
    term_order = ['result', 'args']

    def on_changed(self, which):
        for argname in set(which).intersection(set(self.args.keys())):
            if not self.args[argname].x is getattr(self, argname) :
                self.args[argname].x = getattr(self, argname)
    
    # right now the entries in args have to refer to terms/dterms of result,
    # it would be better if they could be "internal" as well, but for now the idea
    # is that result may itself be a ChLambda.
    def __init__(self, result, args):
        self.args = { argname: ChHandle(x=arg) for argname, arg in list(args.items()) }
        for argname, arg in list(self.args.items()):
            setattr(result, argname, arg)
            if result.is_dr_wrt(arg.x):
                self.add_dterm(argname, arg.x)
            else:
                self.terms.append(argname)
                setattr(self, argname, arg.x)
        self._result = result
        
    def compute_r(self):
        return self._result.r
       
    def compute_dr_wrt(self, wrt):
        return self._result.dr_wrt(wrt)

from .ch_ops import *
from .ch_ops import __all__ as all_ch_ops
__all__ += all_ch_ops

from .reordering import *
from .reordering import Permute
from .reordering import __all__ as all_reordering
__all__ += all_reordering


from . import linalg
from . import ch_random as random
__all__ += ['linalg', 'random']





class tst(Ch):
    dterms = ['a', 'b', 'c']
    
    def compute_r(self):
        return self.a.r + self.b.r + self.c.r
        
    def compute_dr_wrt(self, wrt):
        return 1

def main():
    foo = tst
    
    x10 = Ch(10)
    x20 = Ch(20)
    x30 = Ch(30)
    
    tmp = ChLambda(lambda x, y, z: Ch(1) + Ch(2) * Ch(3) + 4)
    print(tmp.dr_wrt(tmp.x))
    import pdb; pdb.set_trace()
    #a(b(c(d(e(f),g),h)))
    
    blah = tst(x10, x20, x30)
    
    print(blah.r)


    print(foo)
    
    import pdb; pdb.set_trace()
    
    # import unittest
    # from test_ch import TestCh
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestCh)
    # unittest.TextTestRunner(verbosity=2).run(suite)
        


if __name__ == '__main__':
    main()

