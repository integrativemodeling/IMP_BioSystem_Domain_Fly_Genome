import unittest
import subprocess
import sys
import os
import warnings

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

        self.assertTrue(os.path.isfile(PATH+"/data/"+self.crm+".mat"), "Missing hic data.")
        
        p = subprocess.check_call(["python", "01_model_and_analyze.py","--cfg",PATH+"/data/"+self.crm + ".cfg","--ncpus", "12"])
        # check sha
        correct_sha = "0b42fb54-7188-5b47-a922-63b5db43ae42"
        with open(PATH+"/outputs/chr"+self.name+"/models/models.json") as fh:
            sha_lines = [x for x in fh.readlines() if x.lstrip().startswith('"uuid"')]
        sha = ((sha_lines[0]).lstrip())[9:45]
        self.assertEqual(sha, correct_sha)
        if CHKTIME:
            print '12', time() - t0

       
if __name__ == "__main__":
    from time import time
    if len(sys.argv) > 1:
        CHKTIME = bool(int(sys.argv.pop()))
    else:
        CHKTIME = True


    unittest.main()