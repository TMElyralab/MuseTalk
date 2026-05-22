#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['minimize']

import numpy as np
from . import ch
import scipy.sparse as sp
import scipy.optimize

from .optimization_internal import minimize_dogleg

#from memory_profiler import profile, memory_usage

# def disable_cache_for_single_parent_node(node):
#     if hasattr(node, '_parents') and len(node._parents.keys()) == 1:
#         node.want_cache = False


# Nelder-Mead
# Powell
# CG
# BFGS
# Newton-CG
# Anneal
# L-BFGS-B
# TNC
# COBYLA
# SLSQP
# dogleg
# trust-ncg
def minimize(fun, x0, method='dogleg', bounds=None, constraints=(), tol=None, callback=None, options=None):

    if method == 'dogleg':
        if options is None: options = {}
        return minimize_dogleg(fun, free_variables=x0, on_step=callback, **options)

    if isinstance(fun, list) or isinstance(fun, tuple):
        fun = ch.concatenate([f.ravel() for f in fun])
    if isinstance(fun, dict):
        fun = ch.concatenate([f.ravel() for f in list(fun.values())])
    obj = fun
    free_variables = x0


    from .ch import SumOfSquares

    hessp = None
    hess = None
    if obj.size == 1:
        obj_scalar = obj
    else:
        obj_scalar = SumOfSquares(obj)
    
        def hessp(vs, p,obj, obj_scalar, free_variables):
            changevars(vs,obj,obj_scalar,free_variables)
            if not hasattr(hessp, 'vs'):
                hessp.vs = vs*0+1e16
            if np.max(np.abs(vs-hessp.vs)) > 0:

                J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                hessp.J = J
                hessp.H = 2. * J.T.dot(J)
                hessp.vs = vs
            return np.array(hessp.H.dot(p)).ravel()
            #return 2*np.array(hessp.J.T.dot(hessp.J.dot(p))).ravel()
            
        if method.lower() != 'newton-cg':
            def hess(vs, obj, obj_scalar, free_variables):
                changevars(vs,obj,obj_scalar,free_variables)
                if not hasattr(hessp, 'vs'):
                    hessp.vs = vs*0+1e16
                if np.max(np.abs(vs-hessp.vs)) > 0:
                    J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                    hessp.H = 2. * J.T.dot(J)
                return hessp.H
        
    def changevars(vs, obj, obj_scalar, free_variables):
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size
            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz
            
        methods_without_callback = ('anneal', 'powell', 'cobyla', 'slsqp')
        if callback is not None and changed and method.lower() in methods_without_callback:
            callback(None)

        return changed
    
    def residuals(vs,obj, obj_scalar, free_variables):
        changevars(vs, obj, obj_scalar, free_variables)
        residuals = obj_scalar.r.ravel()[0]
        return residuals

    def scalar_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(scalar_jacfunc, 'vs'):
            scalar_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-scalar_jacfunc.vs)) == 0:
            return scalar_jacfunc.J
            
        changevars(vs, obj, obj_scalar, free_variables)
        
        if True: # faster, at least on some problems
            result = np.concatenate([np.array(obj_scalar.lop(wrt, np.array([[1]]))).ravel() for wrt in free_variables])            
        else:
            jacs = [obj_scalar.dr_wrt(wrt) for wrt in free_variables]
            for idx, jac in enumerate(jacs):
                if sp.issparse(jac):
                    jacs[idx] = jacs[idx].todense()
            result = np.concatenate([jac.ravel() for jac in jacs])

        scalar_jacfunc.J = result
        scalar_jacfunc.vs = vs
        return result.ravel()
        
    def ns_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(ns_jacfunc, 'vs'):
            ns_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-ns_jacfunc.vs)) == 0:
            return ns_jacfunc.J
            
        changevars(vs, obj, obj_scalar, free_variables)
        jacs = [obj.dr_wrt(wrt) for wrt in free_variables]
        result = hstack(jacs)

        ns_jacfunc.J = result
        ns_jacfunc.vs = vs
        return result

        
    x1 = scipy.optimize.minimize(
        method=method,
        fun=residuals,
        callback=callback,
        x0=np.concatenate([free_variable.r.ravel() for free_variable in free_variables]),
        jac=scalar_jacfunc,
        hessp=hessp, hess=hess, args=(obj, obj_scalar, free_variables),
        bounds=bounds, constraints=constraints, tol=tol, options=options).x

    changevars(x1, obj, obj_scalar, free_variables)
    return free_variables


def main():
    pass


if __name__ == '__main__':
    main()

