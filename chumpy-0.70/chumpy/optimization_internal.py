import sys
import warnings
import numpy as np
import scipy.sparse as sp
from . import ch, utils
from .ch import pif
from .utils import timer


def clear_cache_single(node):
    node._cache['drs'].clear()
    if hasattr(node, 'dr_cached'):
        node.dr_cached.clear()

def vstack(x):
    x = [a if not isinstance(a, sp.linalg.interface.LinearOperator) else a.dot(np.eye(a.shape[1])) for a in x]
    return sp.vstack(x, format='csc') if any([sp.issparse(a) for a in x]) else np.vstack(x)
def hstack(x):
    x = [a if not isinstance(a, sp.linalg.interface.LinearOperator) else a.dot(np.eye(a.shape[1])) for a in x]
    return sp.hstack(x, format='csc') if any([sp.issparse(a) for a in x]) else np.hstack(x)


_giter = 0
class ChInputsStacked(ch.Ch):
    dterms = 'x', 'obj'
    terms = 'free_variables'

    def compute_r(self):
        if not hasattr(self, 'fevals'):
            self.fevals = 0
        self.fevals += 1
        return self.obj.r.ravel()

    def dr_wrt(self, wrt, profiler=None):
        '''
        Loop over free variables and delete cache for the whole tree after finished each one
        '''
        if wrt is self.x:
            jacs = []
            for fvi, freevar in enumerate(self.free_variables):
                tm = timer()
                if isinstance(freevar, ch.Select):
                    new_jac = self.obj.dr_wrt(freevar.a, profiler=profiler)
                    try:
                        new_jac = new_jac[:, freevar.idxs]
                    except:
                        # non-csc sparse matrices may not support column-wise indexing
                        new_jac = new_jac.tocsc()[:, freevar.idxs]
                else:
                    new_jac = self.obj.dr_wrt(freevar, profiler=profiler)

                pif('dx wrt {} in {}sec, sparse: {}'.format(freevar.short_name, tm(), sp.issparse(new_jac)))

                if self._make_dense and sp.issparse(new_jac):
                    new_jac = new_jac.todense()
                if self._make_sparse and not sp.issparse(new_jac):
                    new_jac = sp.csc_matrix(new_jac)

                if new_jac is None:
                    raise Exception(
                        'Objective has no derivative wrt free variable {}. '
                        'You should likely remove it.'.format(fvi))

                jacs.append(new_jac)
            tm = timer()
            utils.dfs_do_func_on_graph(self.obj, clear_cache_single)
            pif('dfs_do_func_on_graph in {}sec'.format(tm()))
            tm = timer()
            J = hstack(jacs)
            pif('hstack in {}sec'.format(tm()))
            return J

    def on_changed(self, which):
        global _giter
        _giter += 1
        if 'x' in which:
            pos = 0
            for idx, freevar in enumerate(self.free_variables):
                sz = freevar.r.size
                rng = np.arange(pos, pos+sz)
                if isinstance(self.free_variables[idx], ch.Select):
                    # Deal with nested selects
                    selects = []
                    a = self.free_variables[idx]
                    while isinstance(a, ch.Select):
                        selects.append(a.idxs)
                        a = a.a
                    newv = a.x.copy()
                    idxs = selects.pop()
                    while len(selects) > 0:
                        idxs = idxs[selects.pop()]
                    newv.ravel()[idxs] = self.x.r.ravel()[rng]
                    a.__setattr__('x', newv, _giter)
                elif isinstance(self.free_variables[idx].x, np.ndarray):
                    self.free_variables[idx].__setattr__('x', self.x.r[rng].copy().reshape(self.free_variables[idx].x.shape), _giter)
                else: # a number
                    self.free_variables[idx].__setattr__('x', self.x.r[rng], _giter)
                pos += sz

    @property
    def J(self):
        '''
        Compute Jacobian. Analyze dr graph first to disable unnecessary caching
        '''
        result = self.dr_wrt(self.x, profiler=self.profiler).copy()
        if self.profiler:
            self.profiler.harvest()
        return np.atleast_2d(result) if not sp.issparse(result) else result


def setup_sparse_solver(sparse_solver):
    _solver_fns = {
        'cg': lambda A, x, M=None : sp.linalg.cg(A, x, M=M, tol=1e-10)[0],
        'spsolve': lambda A, x : sp.linalg.spsolve(A, x)
    }
    if callable(sparse_solver):
        return sparse_solver
    elif isinstance(sparse_solver, str) and sparse_solver in list(_solver_fns.keys()):
        return _solver_fns[sparse_solver]
    else:
        raise Exception('sparse_solver argument must be either a string in the set (%s) or have the api of scipy.sparse.linalg.spsolve.' % ', '.join(list(_solver_fns.keys())))


def setup_objective(obj, free_variables, on_step=None, disp=True, make_dense=False):
    '''
    obj here can be a list of ch objects or a dict of label: ch objects. Either way, the ch
    objects will be merged into one objective using a ChInputsStacked. The labels are just used
    for printing out values per objective with each iteration. If make_dense is True, the
    resulting object with return a desne Jacobian
    '''
    # Validate free variables
    num_unique_ids = len(np.unique(np.array([id(freevar) for freevar in free_variables])))
    if num_unique_ids != len(free_variables):
        raise Exception('The "free_variables" param contains duplicate variables.')
    # Extract labels
    labels = {}
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = ch.concatenate([f.ravel() for f in obj])
    elif isinstance(obj, dict):
        labels = obj
        obj = ch.concatenate([f.ravel() for f in list(obj.values())])
    # build objective
    x = np.concatenate([freevar.r.ravel() for freevar in free_variables])
    obj = ChInputsStacked(obj=obj, free_variables=free_variables, x=x, make_dense=make_dense)
    # build callback
    def callback():
        if on_step is not None:
            on_step(obj)
        if disp:
            report_line = ['%.2e' % (np.sum(obj.r**2),)]
            for label, objective in sorted(list(labels.items()), key=lambda x: x[0]):
                report_line.append('%s: %.2e' % (label, np.sum(objective.r**2)))
            report_line = " | ".join(report_line) + '\n'
            sys.stderr.write(report_line)
    return obj, callback


class DoglegState(object):
    '''
    Dogleg preserves a great deal of state from iteration to iteration. Many of the things
    that we need to calculate are dependent only on this state (e.g. the various trust region
    steps, the current jacobian and the A & g that depends on it, etc.). Holding the state and
    the various methods based on that state here allows us to seperate a lot of the jacobian
    based calculation from the flow control of the optmization.

    There will be once instance of DoglegState per invocation of minimize_dogleg.
    '''
    def __init__(self, delta, solve):
        self.iteration = 0
        self._d_gn = None # gauss-newton
        self._d_sd = None # steepest descent
        self._d_dl = None # dogleg
        self.J = None
        self.A = None
        self.g = None
        self._p = None
        self.delta = delta
        self.solve = solve
        self._r = None
        self.rho = None
        self.done = False

    @property
    def p(self):
        '''p is the current proposed input vector'''
        return self._p
    @p.setter
    def p(self, val):
        self._p = val.reshape((-1, 1))

    # induce some certainty about what the shape of the steps are
    @property
    def d_gn(self):
        return self._d_gn
    @d_gn.setter
    def d_gn(self, val):
        if val is not None:
            val = val.reshape((-1, 1))
        self._d_gn = val

    @property
    def d_sd(self):
        return self._d_sd
    @d_sd.setter
    def d_sd(self, val):
        if val is not None:
            val = val.reshape((-1, 1))
        self._d_sd = val

    @property
    def d_dl(self):
        return self._d_dl
    @d_dl.setter
    def d_dl(self, val):
        if val is not None:
            val = val.reshape((-1, 1))
        self._d_dl = val

    @property
    def step(self):
        return self.d_dl.reshape((-1, 1))
    @property
    def step_size(self):
        return np.linalg.norm(self.d_dl)

    def start_iteration(self):
        self.iteration += 1
        pif('beginning iteration %d' % (self.iteration,))
        self.d_sd = (np.linalg.norm(self.g)**2 / np.linalg.norm(self.J.dot(self.g))**2 * self.g).ravel()
        self.d_gn = None

    @property
    def r(self):
        '''r is the residual at the current p'''
        return self._r
    @r.setter
    def r(self, val):
        self._r = val.copy().reshape((-1, 1))
        self.updateAg()

    def updateAg(self):
        tm = timer()
        pif('updating A and g...')
        JT = self.J.T
        self.A = JT.dot(self.J)
        self.g = JT.dot(-self.r).reshape((-1, 1))
        pif('A and g updated in %.2fs' % tm())

    def update_step(self):
        # if the Cauchy point is outside the trust region,
        # take that direction but only to the edge of the trust region
        if self.delta is not None and np.linalg.norm(self.d_sd) >= self.delta:
            pif('PROGRESS: Using stunted cauchy')
            self.d_dl = np.array(self.delta/np.linalg.norm(self.d_sd) * self.d_sd).ravel()
        else:
            if self.d_gn is None:
                # We only need to compute this once per iteration
                self.updateGN()
            # if the gauss-newton solution is within the trust region, use it
            if self.delta is None or np.linalg.norm(self.d_gn) <= self.delta:
                pif('PROGRESS: Using gauss-newton solution')
                self.d_dl = np.array(self.d_gn).ravel()
                if self.delta is None:
                    self.delta = np.linalg.norm(self.d_gn)
            else: # between cauchy step and gauss-newton step
                pif('PROGRESS: between cauchy and gauss-newton')
                # apply step
                self.d_dl = self.d_sd + self.beta_multiplier * (self.d_gn - self.d_sd)

    @property
    def beta_multiplier(self):
        delta_sq = self.delta**2
        diff = self.d_gn - self.d_sd
        sqnorm_sd = np.linalg.norm(self.d_sd)**2
        pnow = diff.T.dot(diff)*delta_sq + self.d_gn.T.dot(self.d_sd)**2 - np.linalg.norm(self.d_gn)**2 * sqnorm_sd
        return float(delta_sq - sqnorm_sd) / float((diff).T.dot(self.d_sd) + np.sqrt(pnow))

    def updateGN(self):
        tm = timer()
        if sp.issparse(self.A):
            self.A.eliminate_zeros()
            pif('sparse solve...sparsity infill is %.3f%% (hessian %dx%d)' % (100. * self.A.nnz / (self.A.shape[0] * self.A.shape[1]), self.A.shape[0], self.A.shape[1]))
            if self.g.size > 1:
                self.d_gn = self.solve(self.A, self.g).ravel()
                if np.any(np.isnan(self.d_gn)) or np.any(np.isinf(self.d_gn)):
                    from scipy.sparse.linalg import lsqr
                    warnings.warn("sparse solve failed, falling back to lsqr")
                    self.d_gn = lsqr(self.A, self.g)[0].ravel()
            else:
                self.d_gn = np.atleast_1d(self.g.ravel()[0]/self.A[0,0])
            pif('sparse solve...done in %.2fs' % tm())
        else:
            pif('dense solve...')
            try:
                self.d_gn = np.linalg.solve(self.A, self.g).ravel()
            except Exception:
                warnings.warn("dense solve failed, falling back to lsqr")
                self.d_gn = np.linalg.lstsq(self.A, self.g)[0].ravel()
            pif('dense solve...done in %.2fs' % tm())

    def updateJ(self, obj):
        tm = timer()
        pif('computing Jacobian...')
        self.J = obj.J
        if self.J is None:
            raise Exception("Computing Jacobian failed!")
        if sp.issparse(self.J):
            tm2 = timer()
            self.J = self.J.tocsr()
            pif('converted to csr in {}secs'.format(tm2()))
            assert(self.J.nnz > 0)
        elif ch.VERBOSE:
            nonzero = np.count_nonzero(self.J)
            pif('Jacobian dense with sparsity %.3f' % (nonzero/self.J.size))
        pif('Jacobian (%dx%d) computed in %.2fs' % (self.J.shape[0], self.J.shape[1], tm()))
        if self.J.shape[1] != self.p.size:
            raise Exception('Jacobian size mismatch with objective input')
        return self.J

    class Trial(object):
        '''
        Inside each iteration of dogleg we propose a step and check to see if it's actually
        an improvement before we accept it. This class encapsulates that trial and the
        testing to see if it is actually an improvement.

        There will be one instance of Trial per iteration in dogleg.
        '''
        def __init__(self, proposed_r, state):
            self.r = proposed_r
            self.state = state
            # rho is the ratio of...
            # (improvement in SSE) / (predicted improvement in SSE)
            self.rho = np.linalg.norm(state.r)**2 - np.linalg.norm(proposed_r)**2
            if self.rho > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',category=RuntimeWarning)
                    predicted_improvement = 2. * state.g.T.dot(state.d_dl) - state.d_dl.T.dot(state.A.dot(state.d_dl))
                    self.rho /= predicted_improvement

        @property
        def is_improvement(self):
            return self.rho > 0

        @property
        def improvement(self):
            return (np.linalg.norm(self.state.r)**2 - np.linalg.norm(self.r)**2) / np.linalg.norm(self.state.r)**2

    def trial_r(self, proposed_r):
        return self.Trial(proposed_r, self)

    def updateRadius(self, rho, lb=.05, ub=.9):
        if rho > ub:
            self.delta = max(self.delta, 2.5*np.linalg.norm(self.d_dl))
        elif rho < lb:
            self.delta *= .25


def minimize_dogleg(obj, free_variables, on_step=None,
                     maxiter=200, max_fevals=np.inf, sparse_solver='spsolve',
                     disp=True, e_1=1e-15, e_2=1e-15, e_3=0., delta_0=None,
                     treat_as_dense=False):
    """"Nonlinear optimization using Powell's dogleg method.
    See Lourakis et al, 2005, ICCV '05, "Is Levenberg-Marquardt the
    Most Efficient Optimization for Implementing Bundle Adjustment?":
    http://www.ics.forth.gr/cvrl/publications/conferences/0201-P0401-lourakis-levenberg.pdf

    e_N are stopping conditions:
    e_1 is gradient magnatude threshold
    e_2 is step size magnatude threshold
    e_3 is improvement threshold (as a ratio; 0.1 means it must improve by 10%% at each step)

    maxiter and max_fevals are also stopping conditions. Note that they're not quite the same,
    as an iteration may evaluate the function more than once.

    sparse_solver is the solver to use to calculate the Gauss-Newton step in the common case
    that the Jacobian is sparse. It can be 'spsolve' (in which case scipy.sparse.linalg.spsolve
    will be used), 'cg' (in which case scipy.sparse.linalg.cg will be used), or any callable
    that matches the api of scipy.sparse.linalg.spsolve to solve `A x = b` for x where A is sparse.

    cg, uses a Conjugate Gradient method, and will be faster if A is sparse but x is dense.
    spsolve will be faster if x is also sparse.

    delta_0 defines the initial trust region. Generally speaking, if this is set too low then
    the optimization will never really go anywhere (to small a trust region to make any real
    progress before running out of iterations) and if it's set too high then the optimization
    will diverge immidiately and go wild (such a large trust region that the initial step so
    far overshoots that it can't recover). If it's left as None, it will be automatically
    estimated on the first iteration; it's always updated at each iteration, so this is treated
    only as an initialization.

    handle_as_dense explicitly converts all Jacobians of obj to dense matrices
    """


    solve = setup_sparse_solver(sparse_solver)
    obj, callback = setup_objective(obj, free_variables, on_step=on_step, disp=disp,
                                    make_dense=treat_as_dense)

    state = DoglegState(delta=delta_0, solve=solve)
    state.p = obj.x.r

    #inject profiler if in DEBUG mode
    if ch.DEBUG:
        from .monitor import DrWrtProfiler
        obj.profiler = DrWrtProfiler(obj)

    callback()
    state.updateJ(obj)
    state.r = obj.r

    def stop(msg):
        if not state.done:
            pif(msg)
        state.done = True

    if np.linalg.norm(state.g, np.inf) < e_1:
        stop('stopping because norm(g, np.inf) < %.2e' % e_1)
    while not state.done:
        state.start_iteration()
        while True:
            state.update_step()
            if state.step_size <= e_2 * np.linalg.norm(state.p):
                stop('stopping because of small step size (norm_dl < %.2e)' % (e_2 * np.linalg.norm(state.p)))
            else:
                tm = timer()
                obj.x = state.p + state.step
                trial = state.trial_r(obj.r)
                pif('Residuals computed in %.2fs' % tm())
                # if the objective function improved, update input parameter estimate.
                # Note that the obj.x already has the new parms,
                # and we should not set them again to the same (or we'll bust the cache)
                if trial.is_improvement:
                    state.p = state.p + state.step
                    callback()
                    if e_3 > 0. and trial.improvement < e_3:
                        stop('stopping because improvement < %.1e%%' % (100*e_3))
                    else:
                        state.updateJ(obj)
                        state.r = trial.r
                        if np.linalg.norm(state.g, np.inf) < e_1:
                            stop('stopping because norm(g, np.inf) < %.2e' % e_1)
                else:  # Put the old parms back
                    obj.x = ch.Ch(state.p)
                    obj.on_changed('x') # copies from flat vector to free variables
                # update our trust region
                state.updateRadius(trial.rho)
                if state.delta <= e_2*np.linalg.norm(state.p):
                    stop('stopping because trust region is too small')
            if state.done or trial.is_improvement or (obj.fevals >= max_fevals):
                break
        if state.iteration >= maxiter:
            stop('stopping because max number of user-specified iterations (%d) has been met' % maxiter)
        elif obj.fevals >= max_fevals:
            stop('stopping because max number of user-specified func evals (%d) has been met' % max_fevals)
    return obj.free_variables
