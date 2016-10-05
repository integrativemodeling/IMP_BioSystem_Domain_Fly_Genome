import unittest
import subprocess
import sys
import os
import warnings
import numpy as np
from scipy.stats import spearmanr

PATH = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..'))

class TestTadbit(unittest.TestCase):
    """
    test IMP tadbit functions
    """
    exp = "SRR398"
    reso = "10000"
    name = "4_6_108"
    crm  = "chr4"
    bini  = "6"
    bend  = "108"
    
    def test_1_3d_modelling_optimization(self):
        
        if CHKTIME:
            t0 = time()

        try:
            __import__('IMP')
        except ImportError:
            warnings.warn('IMP not found, skipping test\n')
            return
        os.chdir(os.path.join(PATH, 'scripts'))
        
        name = '{0}_{1}_{2}'.format(self.crm, self.bini, self.bend)
        
        self.assertTrue(os.path.isfile(PATH+"/data/"+self.crm+".mat"), "Missing hic data.")
        
        p = subprocess.check_call(["python", "01_model_and_analyze.py","--cfg",PATH+"/data/"+self.crm + ".cfg","--ncpus", "1"])
        
        # check correlation with real data
        orig_data = np.loadtxt(PATH+"/data/"+self.crm+".mat", delimiter='\t')
        final_data = np.loadtxt(PATH+"/outputs/"+name+"/models/contact_matrix.tsv", delimiter='\t')
        corr = spearmanr(np.array(orig_data).flatten(),np.array(final_data).flatten())
        
        self.assertTrue(corr[0] > 0.75)
        if CHKTIME:
            print '12', time() - t0

       
if __name__ == "__main__":
    from time import time
    if len(sys.argv) > 1:
        CHKTIME = bool(int(sys.argv.pop()))
    else:
        CHKTIME = True


    unittest.main()
