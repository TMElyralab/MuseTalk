#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import time

import unittest
import numpy as np
import scipy.sparse as sp

from . import ch

class TestCh(unittest.TestCase):
    
    
    def test_cachehits(self):
        """Test how many nodes are visited when cache is cleared. 
        If the number of hits changes, it has to be carefully
        looked at to make sure that correctness and performance
        don't get messed up by a change."""
        
        a = ch.array(1)
        b = ch.array(2)
        c = a
        for i in range(10):
            c = a + c + b
    
        c.dr_wrt(a)
        c.dr_wrt(b)
        self.assertEqual(a.clear_cache() + b.clear_cache(), 59)
        c.dr_wrt(a)
        c.dr_wrt(b)
        self.assertEqual(a.clear_cache(123) + b.clear_cache(123), 41)
    
    def test_nested_concatenate(self):
        aa = ch.arange(3)
        bb = ch.arange(4)
        cc = ch.arange(5)
        
        result = ch.concatenate((ch.concatenate((aa,bb)),cc))
        self.assertTrue(result.m0 is aa)
        self.assertTrue(result.m1 is bb)
        self.assertTrue(result.m2 is cc)

        self.assertTrue(result.dr_wrt(aa).nnz > 0)
        self.assertTrue(result.dr_wrt(bb).nnz > 0)
        self.assertTrue(result.dr_wrt(cc).nnz > 0)

    def test_nandivide(self):
        foo = ch.array(np.random.randn(16).reshape((4,4)))
        bar = ch.array(np.random.randn(16).reshape((4,4)))
        bar[2,2] = 0
        self.assertEqual(ch.NanDivide(foo,bar)[2,2].r, 0.)
        foo[2,2] = 0
        self.assertEqual(ch.NanDivide(foo,bar)[2,2].r, 0.)

    def test_casting(self):
        for fn in float, int:
            self.assertEqual(fn(np.array(5)),     fn(ch.array(5)))
            self.assertEqual(fn(np.array([[5]])), fn(ch.array([[5]])))

    def test_tensordot(self):
        an = np.arange(60.).reshape(3,4,5)
        bn = np.arange(24.).reshape(4,3,2)
        cn = np.tensordot(an,bn, axes=([1,0],[0,1]))        
    
        ac = ch.arange(60.).reshape(3,4,5)
        bc = ch.arange(24.).reshape(4,3,2)
        cc = ch.tensordot(ac,bc, axes=([1,0],[0,1]))  
        
        cc.r
        cc.dr_wrt(ac)    
        cc.dr_wrt(bc)    
        #print cn

    def test_make_sure_is_double(self):
        x = ch.array([0])
        self.assertTrue(isinstance(x.r[0], np.float64))

    def test_cross(self):
        aa = ch.random.randn(30).reshape((10,3))
        bb = ch.random.randn(30).reshape((10,3))

        cross_ch = ch.cross(aa, bb)
        cross_np = np.cross(aa.r, bb.r)
        
        # print cross_ch.r
        # print cross_np
        
        eps = 1.0
        step = (np.random.rand(30) - .5).reshape((10,3)) * eps
        
        gt_diff = np.cross(aa.r, bb.r+step) - cross_np
        pr_diff = cross_ch.dr_wrt(bb).dot(step.ravel())
        # print gt_diff
        # print pr_diff
        # print np.max(np.abs(gt_diff.ravel()-pr_diff.ravel()))
        self.assertTrue(1e-14 > np.max(np.abs(gt_diff.ravel()-pr_diff.ravel())))
        
        gt_diff = np.cross(aa.r+step, bb.r) - cross_np
        pr_diff = cross_ch.dr_wrt(aa).dot(step.ravel())
        #print gt_diff
        # print pr_diff
        # print np.max(np.abs(gt_diff.ravel()-pr_diff.ravel()))
        self.assertTrue(1e-14 > np.max(np.abs(gt_diff.ravel()-pr_diff.ravel())))

    def test_dr_wrt_selection(self):
        aa = ch.arange(10,20)
        bb = ch.arange(1,11)
        cc = aa * bb + aa + bb +2
        
        dr0 = cc.dr_wrt(aa[4:6])
        dr1 = cc.dr_wrt(aa)[:,4:6]        
        self.assertTrue((dr0 - dr1).nnz == 0)

        dr0 = cc.dr_wrt(bb[5:8])
        dr1 = cc.dr_wrt(bb)[:,5:8]
        self.assertTrue((dr0 - dr1).nnz == 0)
        

    def test_sum_mean_std_var(self):
        for fn in [ch.sum, ch.mean, ch.var, ch.std]:

            # Create fake input and differences in input space
            data1 = ch.ones((3,4,7,2))
            data2 = ch.array(data1.r + .1 * np.random.rand(data1.size).reshape(data1.shape))
            diff = data2.r - data1.r

            # Compute outputs
            result1 = fn(data1, axis=2)
            result2 = fn(data2, axis=2)

            # Empirical and predicted derivatives
            gt = result2.r - result1.r
            pred = result1.dr_wrt(data1).dot(diff.ravel()).reshape(gt.shape)

            #print np.max(np.abs(gt - pred))

            if fn in [ch.std, ch.var]:
                self.assertTrue(1e-2 > np.max(np.abs(gt - pred)))
            else:
                self.assertTrue(1e-14 > np.max(np.abs(gt - pred)))
                # test caching
                dr0 = result1.dr_wrt(data1)
                data1[:] = np.random.randn(data1.size).reshape(data1.shape)
                self.assertTrue(result1.dr_wrt(data1) is dr0) # changing values shouldn't force recompute
                result1.axis=1
                self.assertTrue(result1.dr_wrt(data1) is not dr0)

        self.assertEqual(ch.mean(ch.eye(3),axis=1).ndim, np.mean(np.eye(3),axis=1).ndim)
        self.assertEqual(ch.mean(ch.eye(3),axis=0).ndim, np.mean(np.eye(3),axis=0).ndim)
        self.assertEqual(ch.sum(ch.eye(3),axis=1).ndim, np.sum(np.eye(3),axis=1).ndim)
        self.assertEqual(ch.sum(ch.eye(3),axis=0).ndim, np.sum(np.eye(3),axis=0).ndim)
            
        

    def test_cumsum(self):
        a = ch.array([1.,5.,3.,7.])
        cs = ch.cumsum(a)
        r1 = cs.r
        dr = cs.dr_wrt(a)
        diff = (ch.random.rand(4)-.5)*.1
        a.x += diff.r
        pred = dr.dot(diff.r)
        gt = cs.r - r1
        self.assertTrue(1e-13 > np.max(np.abs(gt - pred)))
        

    def test_iteration_cache(self):
        """ Each time you set an attribute, the cache (of r's and dr's) of 
        ancestors is cleared. Because children share ancestors, this means
        these can be cleared multiple times unnecessarily; in some cases, 
        where lots of objects exist, this cache clearing can actually be a bottleneck.
        
        Therefore, the concept of an iteration was added; intended to be used in 
        an optimization setting (see optimization.py) and in the set() method, it 
        avoids such redundant clearing of cache."""
        
        a, b, c = ch.Ch(1), ch.Ch(2), ch.Ch(3)
        x = a+b
        y = x+c
        self.assertTrue(y.r[0]==6)
        
        a.__setattr__('x', 10, 1)
        self.assertTrue(y.r == 15)
        a.__setattr__('x', 100, 1)
        self.assertTrue(y.r == 15) 
        a.__setattr__('x', 100, 2)
        self.assertTrue(y.r == 105)  

        a, b, c = ch.array([1]), ch.array([2]), ch.array([3])
        x = a+b
        y = x+c
        self.assertTrue(y.r[0]==6)
        
        a.__setattr__('x', np.array([10]), 1)
        self.assertTrue(y.r[0] == 15)
        a.__setattr__('x', np.array(100), 1)
        self.assertTrue(y.r[0] == 15) 
        a.__setattr__('x', np.array(100), 2)
        self.assertTrue(y.r[0] == 105)  
        a.__setitem__(list(range(0,1)), np.array(200), 2)
        self.assertTrue(y.r[0] == 105)        
        a.__setitem__(list(range(0,1)), np.array(200), 3)
        self.assertTrue(y.r[0] == 205)        
        


    def test_stacking(self):

        a1 = ch.Ch(np.arange(10).reshape(2,5))
        b1 = ch.Ch(np.arange(20).reshape(4,5))
        c1 = ch.vstack((a1,b1))
        c1_check = np.vstack((a1.r, b1.r))
        residuals1 = (c1_check - c1.r).ravel()
        
        
        a2 = ch.Ch(np.arange(10).reshape(5,2))
        b2 = ch.Ch(np.arange(20).reshape(5,4))
        c2 = ch.hstack((a2,b2))
        c2_check = np.hstack((a2.r, b2.r))
        residuals2 = (c2_check - c2.r).ravel()
        
        self.assertFalse(np.any(residuals1))
        self.assertFalse(np.any(residuals2))
        
        d0 = ch.array(np.arange(60).reshape((10,6)))
        d1 = ch.vstack((d0[:4], d0[4:]))
        d2 = ch.hstack((d1[:,:3], d1[:,3:]))
        tmp = d2.dr_wrt(d0).todense()
        diff = tmp - np.eye(tmp.shape[0])
        self.assertFalse(np.any(diff.ravel()))
        
        

    #def test_drs(self):
    #    a = ch.Ch(2)
    #    b = ch.Ch(3)
    #    c = a * b
    #    print c.dr_wrt(a)
    #    print c.compute_drs_wrt(a).r
    
    @unittest.skip('We are using LinearOperator for this for now. Might change back though.')
    def test_reorder_caching(self):
        a = ch.Ch(np.zeros(8).reshape((4,2)))
        b = a.T
        dr0 = b.dr_wrt(a)
        a.x = a.x + 1.
        dr1 = b.dr_wrt(a)
        self.assertTrue(dr0 is dr1)
        a.x = np.zeros(4).reshape((2,2))
        dr2 = b.dr_wrt(a)
        self.assertTrue(dr2 is not dr1)
    
    def test_transpose(self):
        from .utils import row, col
        from copy import deepcopy
        for which in ('C', 'F'): # test in fortran and contiguous mode
            a = ch.Ch(np.require(np.zeros(8).reshape((4,2)), requirements=which))
            b = a.T
        
            b1 = b.r.copy()
            #dr = b.dr_wrt(a).copy()
            dr = deepcopy(b.dr_wrt(a))
        
            diff = np.arange(a.size).reshape(a.shape)
            a.x = np.require(a.r + diff, requirements=which)
            b2 = b.r.copy()
            
            diff_pred = dr.dot(col(diff)).ravel()
            diff_emp =  (b2 - b1).ravel()
            np.testing.assert_array_equal(diff_pred, diff_emp)             
    

    def test_unary(self):
        fns = [ch.exp, ch.log, ch.sin, ch.arcsin, ch.cos, ch.arccos, ch.tan, ch.arctan, ch.negative, ch.square, ch.sqrt, ch.abs, ch.reciprocal]
        
        eps = 1e-8
        for f in fns:
            
            x0 = ch.Ch(.25)
            x1 = ch.Ch(x0.r+eps)
            
            pred = f(x0).dr_wrt(x0)
            empr = (f(x1).r - f(x0).r) / eps

            # print pred
            # print empr
            if f is ch.reciprocal:
                self.assertTrue(1e-6 > np.abs(pred.ravel()[0] - empr.ravel()[0]))
            else:
                self.assertTrue(1e-7 > np.abs(pred.ravel()[0] - empr.ravel()[0]))
            

    def test_serialization(self):
        # The main challenge with serialization is the "_parents" 
        # attribute, which is a nonserializable WeakKeyDictionary. 
        # So we pickle/unpickle, change a child and verify the value
        # at root, and verify that both children have parentage.
        from six.moves import cPickle as pickle
        tmp = ch.Ch(10) + ch.Ch(20)
        tmp = pickle.loads(pickle.dumps(tmp))
        tmp.b.x = 30
        self.assertTrue(tmp.r[0] == 40)
        self.assertTrue(list(tmp.a._parents.keys())[0] == tmp)
        self.assertTrue(list(tmp.a._parents.keys())[0] == list(tmp.b._parents.keys())[0])
        
    def test_chlambda1(self):
        c1, c2, c3 = ch.Ch(1), ch.Ch(2), ch.Ch(3)
        adder = ch.ChLambda(lambda x, y: x+y)
        adder.x = c1
        adder.y = c2
        self.assertTrue(adder.r == 3)
        adder.x = c2
        self.assertTrue(adder.r == 4)
        adder.x = c1
        self.assertTrue(adder.r == 3)        


    def test_chlambda2(self):
        passthrough = ch.ChLambda( lambda x : x)
        self.assertTrue(passthrough.dr_wrt(passthrough.x) is not None)
        passthrough.x = ch.Ch(123)
        self.assertTrue(passthrough.dr_wrt(passthrough.x) is not None)

    # It's probably not reasonable to expect this
    # to work for ChLambda
    #def test_chlambda3(self):
    #    c1, c2, c3 = ch.Ch(1), ch.Ch(2), ch.Ch(3)    
    #    triple = ch.ChLambda( lambda x, y, z : x(y, z))
    #    triple.x = Add
    #    triple.y = c2
    #    triple.z = c3
    
    

        
    
    def test_amax(self):
        from .ch import amax
        import numpy as np
        arr = np.empty((5,2,3,7))
        arr.flat[:] = np.sin(np.arange(arr.size)*1000.)
        #arr = np.array(np.sin(np.arange(24)*10000.).reshape(2,3,4))
        
        for axis in range(len(arr.shape)):
            a = amax(a=arr, axis=axis)
            pred = a.dr_wrt(a.a).dot(arr.ravel())
            real = np.amax(arr, axis=axis).ravel()
            self.assertTrue(np.max(np.abs(pred-real)) < 1e-10)

    def test_maximum(self):
        from .utils import row, col
        from .ch import maximum
        
        # Make sure that when we compare the max of two *identical* numbers,
        # we get the right derivatives wrt both
        the_max = maximum(ch.Ch(1), ch.Ch(1))
        self.assertTrue(the_max.r.ravel()[0] == 1.)
        self.assertTrue(the_max.dr_wrt(the_max.a)[0,0] == 1.)
        self.assertTrue(the_max.dr_wrt(the_max.b)[0,0] == 1.)
        
        # Now test given that all numbers are different, by allocating from
        # a pool of randomly permuted numbers.
        # We test combinations of scalars and 2d arrays.
        rnd = np.asarray(np.random.permutation(np.arange(20)), np.float64)
        c1 = ch.Ch(rnd[:6].reshape((2,3)))
        c2 = ch.Ch(rnd[6:12].reshape((2,3)))
        s1 = ch.Ch(rnd[12])
        s2 = ch.Ch(rnd[13])
        
        eps = .1
        for first in [c1, s1]:
            for second in [c2, s2]:
                the_max = maximum(first, second)
                
                for which_to_change in [first, second]:
                    
                
                    max_r0 = the_max.r.copy()                
                    max_r_diff = np.max(np.abs(max_r0 - np.maximum(first.r, second.r)))
                    self.assertTrue(max_r_diff == 0)
                    max_dr = the_max.dr_wrt(which_to_change).copy()
                    which_to_change.x = which_to_change.x + eps
                    max_r1 = the_max.r.copy()
                    
                    emp_diff = (the_max.r - max_r0).ravel()
                    pred_diff = max_dr.dot(col(eps*np.ones(max_dr.shape[1]))).ravel()
                    
                    #print 'comparing the following numbers/vectors:'
                    #print first.r
                    #print second.r
                    #print 'empirical vs predicted difference:'
                    #print emp_diff
                    #print pred_diff
                    #print '-----'
                    
                    max_dr_diff = np.max(np.abs(emp_diff-pred_diff))
                    #print 'max dr diff: %.2e' % (max_dr_diff,)
                    self.assertTrue(max_dr_diff < 1e-14)        
            

    def test_shared(self):
    
        chs = [ch.Ch(i) for i in range(10)]
        vrs = [float(i) for i in range(10)]
    
        func = lambda a : a[0]*a[1] + (a[2]*a[3])/a[4]
        
        chained_result = func(chs).r
        regular_result = func(vrs)
        
        self.assertTrue(chained_result == regular_result)
        #print chained_result
        #print regular_result
        
        chained_func = func(chs)
        chained_func.replace(chs[0], ch.Ch(50))
        vrs[0] = 50
        
        chained_result = chained_func.r
        regular_result = func(vrs)
    
        self.assertTrue(chained_result == regular_result)
        #print chained_result
        #print regular_result
            

    def test_matmatmult(self):
        from .ch import dot
        mtx1 = ch.Ch(np.arange(6).reshape((3,2)))
        mtx2 = ch.Ch(np.arange(8).reshape((2,4))*10)
        
        mtx3 = dot(mtx1, mtx2)
        #print mtx1.r
        #print mtx2.r
        #print mtx3.r
        #print mtx3.dr_wrt(mtx1).todense()
        #print mtx3.dr_wrt(mtx2).todense()
        
        for mtx in [mtx1, mtx2]:
            oldval = mtx3.r.copy()
            mtxd = mtx3.dr_wrt(mtx).copy()
            mtx_diff = np.random.rand(mtx.r.size).reshape(mtx.r.shape)
            mtx.x = mtx.r + mtx_diff
            mtx_emp = mtx3.r - oldval
            mtx_pred = mtxd.dot(mtx_diff.ravel()).reshape(mtx_emp.shape)
        
            self.assertTrue(np.max(np.abs(mtx_emp - mtx_pred)) < 1e-11)

    
    def test_ndim(self):
        vs = [ch.Ch(np.random.randn(6).reshape(2,3)) for i in range(6)]        
        res = vs[0] + vs[1] - vs[2] * vs[3] / (vs[4] ** 2) ** vs[5]
        self.assertTrue(res.shape[0]==2 and res.shape[1]==3)
        res = (vs[0] + 1) + (vs[1] - 2) - (vs[2] * 3) * (vs[3] / 4)  / (vs[4] ** 2) ** vs[5]
        self.assertTrue(res.shape[0]==2 and res.shape[1]==3)
        drs = [res.dr_wrt(v) for v in vs]
        

    def test_indexing(self):
        big = ch.Ch(np.arange(60).reshape((10,6)))
        little = big[1:3, 3:6]
        self.assertTrue(np.max(np.abs(little.r - np.array([[9,10,11],[15,16,17]]))) == 0)
        
        little = big[5]
        self.assertTrue(np.max(np.abs(little.r - np.arange(30, 36))) == 0)
        self.assertTrue(np.max(np.abs(sp.coo_matrix(little.dr_wrt(big)).col - np.arange(30,36))) == 0)
        
        little = big[2, 3]
        self.assertTrue(little.r[0] == 15.0)
        
        little = big[2, 3:5]
        self.assertTrue(np.max(np.abs(little.r - np.array([15, 16]))) == 0.)
        _ =  little.dr_wrt(big)

        # Tests assignment through reorderings
        aa = ch.arange(4*4*4).reshape((4,4,4))[:3,:3,:3]
        aa[0,1,2] = 100
        self.assertTrue(aa[0,1,2].r[0] == 100)

        # Tests assignment through reorderings (NaN's are a special case)
        aa = ch.arange(9).reshape((3,3))
        aa[1,1] = np.nan
        self.assertTrue(np.isnan(aa.r[1,1]))
        self.assertFalse(np.isnan(aa.r[0,0]))


    def test_redundancy_removal(self):

        for MT in [False, True]:
            x1, x2 = ch.Ch(10), ch.Ch(20)
            x1_plus_x2_1 = x1 + x2
            x1_plus_x2_2 = x1 + x2
            redundant_sum = (x1_plus_x2_1 + x1_plus_x2_2) * 2
            redundant_sum.MT = MT
        
            self.assertTrue(redundant_sum.a.a is not redundant_sum.a.b)
            redundant_sum.remove_redundancy()
            self.assertTrue(redundant_sum.a.a is redundant_sum.a.b)
        
    def test_caching(self):
            
        vals = [10, 20, 30, 40, 50]
        f = lambda a, b, c, d, e : a + (b * c) - d ** e

        # Set up our objects
        Cs = [ch.Ch(v) for v in vals]
        C_result = f(*Cs)

        # Sometimes residuals should be cached
        r1 = C_result.r
        r2 = C_result.r
        self.assertTrue(r1 is r2)
        
        # Other times residuals need refreshing
        Cs[0].set(x=5)
        r3 = C_result.r
        self.assertTrue(r3 is not r2)
        
        # Sometimes derivatives should be cached
        dr1 = C_result.dr_wrt(Cs[1])
        dr2 = C_result.dr_wrt(Cs[1])
        self.assertTrue(dr1 is dr2)
        
        # Other times derivatives need refreshing
        Cs[2].set(x=5)
        dr3 = C_result.dr_wrt(Cs[1])
        self.assertTrue(dr3 is not dr2)


    def test_scalars(self):
        
        try:
            import theano.tensor as T
            from theano import function            
        except:
            return
        
        # Set up variables and function
        vals = [1, 2, 3, 4, 5]
        f = lambda a, b, c, d, e : a + (b * c) - d ** e

        # Set up our objects
        Cs = [ch.Ch(v) for v in vals]
        C_result = f(*Cs)

        # Set up Theano's equivalents
        Ts = T.dscalars('T1', 'T2', 'T3', 'T4', 'T5')
        TF = f(*Ts)        
        T_result = function(Ts, TF)        

        # Make sure values and derivatives are equal
        self.assertEqual(C_result.r, T_result(*vals))
        for k in range(len(vals)):
            theano_derivative = function(Ts, T.grad(TF, Ts[k]))(*vals)
            #print C_result.dr_wrt(Cs[k])
            our_derivative = C_result.dr_wrt(Cs[k])[0,0]
            #print theano_derivative, our_derivative
            self.assertEqual(theano_derivative, our_derivative)
        

    def test_vectors(self):
        
        try:
            import theano.tensor as T
            from theano import function            
        except:
            return
            
        for MT in [False, True]:

            # Set up variables and function
            vals = [np.random.randn(20) for i in range(5)]
            f = lambda a, b, c, d, e : a + (b * c) - d ** e

            # Set up our objects
            Cs = [ch.Ch(v) for v in vals]
            C_result = f(*Cs)
            C_result.MT = MT

            # Set up Theano equivalents
            Ts = T.dvectors('T1', 'T2', 'T3', 'T4', 'T5')
            TF = f(*Ts)
            T_result = function(Ts, TF)        

            if False:
                import theano.gradient
                which = 1
                theano_sse = (TF**2.).sum()
                theano_grad = theano.gradient.grad(theano_sse, Ts[which])
                theano_fn = function(Ts, theano_grad)
                print(theano_fn(*vals))
                C_result_grad = ch.SumOfSquares(C_result).dr_wrt(Cs[which])
                print(C_result_grad)
                
                # if True:
                #     aaa = np.linalg.solve(C_result_grad.T.dot(C_result_grad), C_result_grad.dot(np.zeros(C_result_grad.shape[1])))
                #     theano_hes = theano.R_obbb = theano.R_op()
                
                import pdb; pdb.set_trace()

            # Make sure values and derivatives are equal
            np.testing.assert_array_equal(C_result.r, T_result(*vals))
            for k in range(len(vals)):
                theano_derivative = function(Ts, T.jacobian(TF, Ts[k]))(*vals)
                our_derivative = np.array(C_result.dr_wrt(Cs[k]).todense())
                #print theano_derivative, our_derivative   
            
                # Theano produces has more nans than we do during exponentiation. 
                # So we test only on entries where Theano is without NaN's    
                without_nans = np.nonzero(np.logical_not(np.isnan(theano_derivative.flatten())))[0]
                np.testing.assert_array_equal(theano_derivative.flatten()[without_nans], our_derivative.flatten()[without_nans])
    
    
if __name__ == '__main__':
    unittest.main()
