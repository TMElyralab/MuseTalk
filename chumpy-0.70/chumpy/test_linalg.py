#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import numpy as np
import unittest

from .ch import Ch




class TestLinalg(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)


    def test_slogdet(self):
        from . import ch
        tmp = ch.random.randn(100).reshape((10,10))
        # print 'chumpy version: ' + str(slogdet(tmp)[1].r)
        # print 'old version:' + str(np.linalg.slogdet(tmp.r)[1])

        eps = 1e-10
        diff = np.random.rand(100) * eps
        diff_reshaped = diff.reshape((10,10))
        gt = np.linalg.slogdet(tmp.r+diff_reshaped)[1] - np.linalg.slogdet(tmp.r)[1]
        pred = ch.linalg.slogdet(tmp)[1].dr_wrt(tmp).dot(diff)
        #print gt
        #print pred
        diff = gt - pred
    
        self.assertTrue(np.max(np.abs(diff)) < 1e-12)

        sgn_gt = np.linalg.slogdet(tmp.r)[0]
        sgn_pred = ch.linalg.slogdet(tmp)[0]

        #print sgn_gt
        #print sgn_pred
        diff = sgn_gt - sgn_pred.r
        self.assertTrue(np.max(np.abs(diff)) < 1e-12)
        

    def test_lstsq(self):
        from .linalg import lstsq
        
        shapes = ([10, 3], [3, 10])
        
        for shape in shapes:
            for b2d in True, False:
                A = (np.random.rand(np.prod(shape))-.5).reshape(shape)
                if b2d:
                    b = np.random.randn(shape[0],2)
                else:
                    b = np.random.randn(shape[0])
        
                x1, residuals1, rank1, s1 = lstsq(A, b)
                x2, residuals2, rank2, s2 = np.linalg.lstsq(A, b)
        
                #print x1.r
                #print x2
                #print residuals1.r
                #print residuals2
                self.assertTrue(np.max(np.abs(x1.r-x2)) < 1e-14)                
                if len(residuals2) > 0:
                    self.assertTrue(np.max(np.abs(residuals1.r-residuals2)) < 1e-14)
            
            
        

    def test_pinv(self):
        from .linalg import Pinv
        
        data = (np.random.rand(12)-.5).reshape((3, 4))
        pc_tall = Pinv(data)
        pc_wide = Pinv(data.T)
        
        pn_tall = np.linalg.pinv(data)
        pn_wide = np.linalg.pinv(data.T)
        
        tall_correct = np.max(np.abs(pc_tall.r - pn_tall)) < 1e-12
        wide_correct = np.max(np.abs(pc_wide.r - pn_wide)) < 1e-12
        # if not tall_correct or not wide_correct:
        #     print tall_correct
        #     print wide_correct
        #     import pdb; pdb.set_trace()
        self.assertTrue(tall_correct)
        self.assertTrue(wide_correct)
        
        return # FIXME. how to test derivs?
        
        for pc in [pc_tall, pc_wide]:
            
            self.chkd(pc, pc.mtx)
            import pdb; pdb.set_trace()
            
            

    def test_svd(self):
        from .linalg import Svd
        eps = 1e-3
        idx = 10

        data = np.sin(np.arange(300)*100+10).reshape((-1,3))
        data[3,:] = data[3,:]*0+10
        data[:,1] *= 2
        data[:,2] *= 4
        data = data.copy()
        u,s,v = np.linalg.svd(data, full_matrices=False)
        data = Ch(data)
        data2 = data.r.copy()
        data2.ravel()[idx] += eps
        u2,s2,v2 = np.linalg.svd(data2, full_matrices=False)


        svdu, svdd, svdv = Svd(x=data)

        # test singular values
        diff_emp = (s2-s) / eps
        diff_pred = svdd.dr_wrt(data)[:,idx]
        #print diff_emp
        #print diff_pred
        ratio = diff_emp / diff_pred
        #print ratio
        self.assertTrue(np.max(np.abs(ratio - 1.)) < 1e-4)
            
        # test V
        diff_emp = (v2 - v) / eps
        diff_pred = svdv.dr_wrt(data)[:,idx].reshape(diff_emp.shape)
        ratio = diff_emp / diff_pred
        #print ratio
        self.assertTrue(np.max(np.abs(ratio - 1.)) < 1e-2)

        # test U
        diff_emp = (u2 - u) / eps
        diff_pred = svdu.dr_wrt(data)[:,idx].reshape(diff_emp.shape)
        ratio = diff_emp / diff_pred
        #print ratio
        self.assertTrue(np.max(np.abs(ratio - 1.)) < 1e-2)
        
        
    def test_det(self):
        from .linalg import Det
        
        mtx1 = Ch(np.sin(2**np.arange(9)).reshape((3,3)))
        mtx1_det = Det(mtx1)
        dr = mtx1_det.dr_wrt(mtx1)

        eps = 1e-5
        mtx2 = mtx1.r.copy()
        input_diff = np.sin(np.arange(mtx2.size)).reshape(mtx2.shape) * eps
        mtx2 += input_diff
        mtx2_det = Det(mtx2)

        output_diff_emp = (np.linalg.det(mtx2) - np.linalg.det(mtx1.r)).ravel()
        
        output_diff_pred = Det(mtx1).dr_wrt(mtx1).dot(input_diff.ravel())

        #print output_diff_emp
        #print output_diff_pred

        self.assertTrue(np.max(np.abs(output_diff_emp - output_diff_pred)) < eps*1e-4)
        self.assertTrue(np.max(np.abs(mtx1_det.r - np.linalg.det(mtx1.r)).ravel()) == 0)
        
    
        
    def test_inv1(self):
        from .linalg import Inv

        mtx1 = Ch(np.sin(2**np.arange(9)).reshape((3,3)))
        mtx1_inv = Inv(mtx1)
        dr = mtx1_inv.dr_wrt(mtx1)

        eps = 1e-5
        mtx2 = mtx1.r.copy()
        input_diff = np.sin(np.arange(mtx2.size)).reshape(mtx2.shape) * eps
        mtx2 += input_diff
        mtx2_inv = Inv(mtx2)

        output_diff_emp = (np.linalg.inv(mtx2) - np.linalg.inv(mtx1.r)).ravel()
        output_diff_pred = Inv(mtx1).dr_wrt(mtx1).dot(input_diff.ravel())

        #print output_diff_emp
        #print output_diff_pred

        self.assertTrue(np.max(np.abs(output_diff_emp - output_diff_pred)) < eps*1e-4)
        self.assertTrue(np.max(np.abs(mtx1_inv.r - np.linalg.inv(mtx1.r)).ravel()) == 0)

    def test_inv2(self):
        from .linalg import Inv
                
        eps = 1e-8
        idx = 13

        mtx1 = np.random.rand(100).reshape((10,10))
        mtx2 = mtx1.copy()
        mtx2.ravel()[idx] += eps
        
        diff_emp = (np.linalg.inv(mtx2) - np.linalg.inv(mtx1)) / eps
        
        mtx1 = Ch(mtx1)
        diff_pred = Inv(mtx1).dr_wrt(mtx1)[:,13].reshape(diff_emp.shape)
        #print diff_emp
        #print diff_pred
        #print diff_emp - diff_pred
        self.assertTrue(np.max(np.abs(diff_pred.ravel()-diff_emp.ravel())) < 1e-4)
        
    @unittest.skipIf(np.__version__ < '1.8',
                     "broadcasting for matrix inverse not supported in numpy < 1.8")
    def test_inv3(self):
        """Test linalg.inv with broadcasting support."""
        
        from .linalg import Inv

        mtx1 = Ch(np.sin(2**np.arange(12)).reshape((3,2,2)))
        mtx1_inv = Inv(mtx1)
        dr = mtx1_inv.dr_wrt(mtx1)

        eps = 1e-5
        mtx2 = mtx1.r.copy()
        input_diff = np.sin(np.arange(mtx2.size)).reshape(mtx2.shape) * eps
        mtx2 += input_diff
        mtx2_inv = Inv(mtx2)

        output_diff_emp = (np.linalg.inv(mtx2) - np.linalg.inv(mtx1.r)).ravel()
        output_diff_pred = Inv(mtx1).dr_wrt(mtx1).dot(input_diff.ravel())

        # print output_diff_emp
        # print output_diff_pred

        self.assertTrue(np.max(np.abs(output_diff_emp.ravel() - output_diff_pred.ravel())) < eps*1e-3)
        self.assertTrue(np.max(np.abs(mtx1_inv.r - np.linalg.inv(mtx1.r)).ravel()) == 0)

    def chkd(self, obj, parm, eps=1e-14):
        backed_up = parm.x

        if True:
            diff = (np.random.rand(parm.size)-.5).reshape(parm.shape)            
        else:
            diff = np.zeros(parm.shape)
            diff.ravel()[4] = 2.

        dr = obj.dr_wrt(parm)

        parm.x = backed_up - diff*eps
        r_lower = obj.r

        parm.x = backed_up + diff*eps
        r_upper = obj.r

        diff_emp = (r_upper - r_lower) / (eps*2.)
        diff_pred = dr.dot(diff.ravel()).reshape(diff_emp.shape)

        #print diff_emp
        #print diff_pred
        print(diff_emp / diff_pred)
        print(diff_emp - diff_pred)

        parm.x = backed_up        

        

suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalg)

if __name__ == '__main__':
    unittest.main()

