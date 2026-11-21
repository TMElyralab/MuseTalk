#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import unittest
from .ch import Ch, depends_on

class TestInnerComposition(unittest.TestCase):

    def test_ic(self):
        child = Child(a=Ch(10))
        parent = Parent(child=child, aliased=Ch(50))
        
        junk = [parent.aliased_dependency for k in range(3)]
        self.assertTrue(parent.dcount == 1)
        self.assertTrue(parent.ocount == 0)
        self.assertTrue(parent.rcount == 0)
        
        junk = [parent.r for k in range(3)]
        self.assertTrue(parent.dcount == 1)
        self.assertTrue(parent.ocount == 1)
        self.assertTrue(parent.rcount == 1)
        
        parent.aliased = Ch(20)
        junk = [parent.aliased_dependency for k in range(3)]
        self.assertTrue(parent.dcount == 2)
        self.assertTrue(parent.ocount == 1)
        self.assertTrue(parent.rcount == 1)
        
        junk = [parent.r for k in range(3)]
        self.assertTrue(parent.dcount == 2)
        self.assertTrue(parent.ocount == 2)
        self.assertTrue(parent.rcount == 2)
        
class Parent(Ch):
    dterms = ('aliased', 'child')
    
    def __init__(self, *args, **kwargs):
        self.dcount = 0
        self.ocount = 0
        self.rcount = 0
    
    
    def on_changed(self, which):
        assert('aliased' in which and 'child' in which)
        if 'aliased' in which:
            self.ocount += 1
        
    @depends_on('aliased')
    def aliased_dependency(self):
        self.dcount += 1        
        
    @property
    def aliased(self):
        return self.child.a
    
    @aliased.setter
    def aliased(self, val):
        self.child.a = val
    
    def compute_r(self):
        self.rcount += 1
        return 0
    
    def compute_dr_wrt(self, wrt):
        pass
    
    
class Child(Ch):
    dterms = ('a',)
    
    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInnerComposition)
    unittest.TextTestRunner(verbosity=2).run(suite)
