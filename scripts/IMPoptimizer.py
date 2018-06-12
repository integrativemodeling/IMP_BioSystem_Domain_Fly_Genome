"""
28 Aug 2013


"""
from math      import log10, fabs, pow as power
from scipy.stats                    import spearmanr
from scipy                         import polyfit
from os.path                       import exists
from sys                           import stdout
from cPickle                       import dump, load
from sys                           import stderr
import numpy           as np
import multiprocessing as mu

import IMP.core
from IMP.container import ListSingletonContainer
from IMP import Model
from IMP import FloatKey


IMP.set_check_level(IMP.NONE)
IMP.set_log_level(IMP.SILENT)

# GENERAL
#########

CONFIG = {
    'dmel_01': {
        # use these paramaters with the Hi-C data from:
        'reference' : 'victor corces dataset 2013',
        
        # Force applied to the restraints inferred to neighbor particles
        'kforce'    : 5,
        
        # Maximum experimental contact distance
        'maxdist'   : 600, # OPTIMIZATION: 500-1200
        
        # Maximum thresholds used to decide which experimental values have to be
        # included in the computation of restraints. Z-score values bigger than upfreq
        # and less that lowfreq will be include, whereas all the others will be rejected
        'upfreq'    : 0.3, # OPTIMIZATION: min/max Z-score
        
        # Minimum thresholds used to decide which experimental values have to be
        # included in the computation of restraints. Z-score values bigger than upfreq
        # and less that lowfreq will be include, whereas all the others will be rejected
        'lowfreq'   : -0.7, # OPTIMIZATION: min/max Z-score

        # How much space (in nm) ocupies a nucleotide
        'scale'     : 0.01
        
        }
    }


# MonteCarlo optimizer parameters
#################################
# number of iterations
NROUNDS   = 10000
# number of MonteCarlo steps per round
STEPS     = 1
# number of local steps per round
LSTEPS    = 5

class IMPoptimizer(object):
    """
    This class optimizes a set of paramaters (scale, maxdist, lowfreq and
    upfreq) in order to maximize the correlation between the models generated 
    by IMP and the input data.

    :param experiment: an instance of the class pytadbit.experiment.Experiment
    :param start: first bin to model (bin number)
    :param end: last bin to model (bin number)
    :param 5000 n_models: number of models to generate
    :param 1000 n_keep: number of models used in the final analysis (usually 
       the top 20% of the generated models). The models are ranked according to
       their objective function value (the lower the better)
    :param 1 close_bins: number of particles away (i.e. the bin number 
       difference) a particle pair must be in order to be considered as 
       neighbors (e.g. 1 means consecutive particles)
    :param None cutoff: distance cutoff (nm) to define whether two particles
       are in contact or not, default is 2 times resolution, times scale.
    :param None container: restrains particle to be within a given object. Can 
       only be a 'cylinder', which is, in fact a cylinder of a given height to 
       which are added hemispherical ends. This cylinder is defined by a radius, 
       its height (with a height of 0 the cylinder becomes a sphere) and the 
       force applied to the restraint. E.g. for modeling E. coli genome (2 
       micrometers length and 0.5 micrometer of width), these values could be 
       used: ['cylinder', 250, 1500, 50], and for a typical mammalian nuclei
       (6 micrometers diameter): ['cylinder', 3000, 0, 50]
    """
    def __init__(self, norm, resolution, start, end, n_models=500,
                 n_keep=100, close_bins=1, container=None):

        self.resolution = resolution
        self.size = len(norm[0])
        self.norm = norm
        self.zscores, self.values, self.zeros = get_hic_zscores(self.norm, self.size)
        self.nloci       = end - start + 1
        
        self.n_models    = n_models
        self.n_keep      = n_keep
        self.close_bins  = close_bins

        self.scale_range   = []
        self.maxdist_range = []
        self.lowfreq_range = []
        self.upfreq_range  = []
        self.dcutoff_range = []
        self.container     = container
        self.results = {}
        self.__models       = []
        self._bad_models    = []
        
        
    
    def run_grid_search(self, 
                        upfreq_range=(0, 1, 0.1),
                        lowfreq_range=(-1, 0, 0.1),
                        maxdist_range=(400, 1500, 100),
                        scale_range=0.01,
                        dcutoff_range=2,
                        corr='spearman', off_diag=1,
                        n_cpus=1, verbose=True):
        """
        This function calculates the correlation between the models generated 
        by IMP and the input data for the four main IMP parameters (scale, 
        maxdist, lowfreq and upfreq) in the given ranges of values.
        
        :param n_cpus: number of CPUs to use
        :param (-1,0,0.1) lowfreq_range: range of lowfreq values to be 
           optimized. The last value of the input tuple is the incremental 
           step for the lowfreq values
        :param (0,1,0.1) upfreq_range: range of upfreq values to be optimized.
           The last value of the input tuple is the incremental step for the
           upfreq values
        :param (400,1400,100) maxdist_range: upper and lower bounds
           used to search for the optimal maximum experimental distance. The 
           last value of the input tuple is the incremental step for maxdist 
           values
        :param 0.01 scale_range: upper and lower bounds used to search for
           the optimal scale parameter (nm per nucleotide). The last value of
           the input tuple is the incremental step for scale parameter values
        :param 2 dcutoff_range: upper and lower bounds used to search for
           the optimal distance cutoff parameter (distance, in number of beads,
           from which to consider 2 beads as being close). The last value of the
           input tuple is the incremental step for scale parameter values
      
        :param True verbose: print the results to the standard output
        """
        if verbose:
            stderr.write('Optimizing %s particles\n' % self.nloci)
        if isinstance(maxdist_range, tuple):
            maxdist_step = maxdist_range[2]
            maxdist_arange = range(maxdist_range[0],
                                        maxdist_range[1] + maxdist_step,
                                        maxdist_step)
        else:
            if isinstance(maxdist_range, (float, int)):
                maxdist_range = [maxdist_range]
            maxdist_arange = maxdist_range
        #
        if isinstance(lowfreq_range, tuple):
            lowfreq_step = lowfreq_range[2]
            lowfreq_arange = np.arange(lowfreq_range[0],
                                            lowfreq_range[1] + lowfreq_step / 2,
                                            lowfreq_step)
        else:
            if isinstance(lowfreq_range, (float, int)):
                lowfreq_range = [lowfreq_range]
            lowfreq_arange = lowfreq_range
        #
        if isinstance(upfreq_range, tuple):
            upfreq_step = upfreq_range[2]
            upfreq_arange = np.arange(upfreq_range[0],
                                           upfreq_range[1] + upfreq_step / 2,
                                           upfreq_step)
        else:
            if isinstance(upfreq_range, (float, int)):
                upfreq_range = [upfreq_range]
            upfreq_arange = upfreq_range
        #
        if isinstance(scale_range, tuple):
            scale_step = scale_range[2]
            scale_arange = np.arange(scale_range[0],
                                          scale_range[1] + scale_step / 2,
                                          scale_step)
        else:
            if isinstance(scale_range, (float, int)):
                scale_range = [scale_range]
            scale_arange = scale_range
        #
        if isinstance(dcutoff_range, tuple):
            dcutoff_step = dcutoff_range[2]
            dcutoff_arange = np.arange(dcutoff_range[0],
                                          dcutoff_range[1] + dcutoff_step / 2,
                                          dcutoff_step)
        else:
            if isinstance(dcutoff_range, (float, int)):
                dcutoff_range = [dcutoff_range]
            dcutoff_arange = dcutoff_range

        # round everything
        if not self.maxdist_range:
            self.maxdist_range = [my_round(i) for i in maxdist_arange]
        else:
            self.maxdist_range = sorted([my_round(i) for i in maxdist_arange
                                         if not my_round(i) in self.maxdist_range] +
                                        self.maxdist_range)
        if not self.upfreq_range:
            self.upfreq_range  = [my_round(i) for i in upfreq_arange ]
        else:
            self.upfreq_range = sorted([my_round(i) for i in upfreq_arange
                                        if not my_round(i) in self.upfreq_range] +
                                       self.upfreq_range)
        if not self.lowfreq_range:
            self.lowfreq_range = [my_round(i) for i in lowfreq_arange]
        else:
            self.lowfreq_range = sorted([my_round(i) for i in lowfreq_arange
                                         if not my_round(i) in self.lowfreq_range] +
                                        self.lowfreq_range)
        if not self.scale_range:
            self.scale_range   = [my_round(i) for i in scale_arange  ]
        else:
            self.scale_range = sorted([my_round(i) for i in scale_arange
                                       if not my_round(i) in self.scale_range] +
                                      self.scale_range)
        if not self.dcutoff_range:
            self.dcutoff_range = [my_round(i) for i in dcutoff_arange]
        else:
            self.dcutoff_range = sorted([my_round(i) for i in dcutoff_arange
                                         if not my_round(i) in self.dcutoff_range] +
                                        self.dcutoff_range)
        # grid search
        models = {}
        count = 0
        if verbose:
            stderr.write('# %3s %6s %7s %7s %6s %7s %7s\n' % (
                                    "num", "upfrq", "lowfrq", "maxdist",
                                    "scale", "cutoff", "corr"))
        for scale in [my_round(i) for i in scale_arange]:
            for maxdist in [my_round(i) for i in maxdist_arange]:
                for upfreq in [my_round(i) for i in upfreq_arange]:
                    for lowfreq in [my_round(i) for i in lowfreq_arange]:
                        # check if this optimization has been already done
                        if (scale, maxdist, upfreq, lowfreq) in [
                            tuple(k[:4]) for k in self.results]:
                            k = [k for k in self.results
                                 if (scale, maxdist, upfreq,
                                     lowfreq) == tuple(k[:4])][0]
                            result = self.results[(scale, maxdist, upfreq,
                                                   lowfreq, k[-1])]
                            if verbose:
                                verb = '%5s %6s %7s %7s %6s %7s  ' % (
                                    'xx', upfreq, lowfreq, maxdist,
                                    scale, k[-1])
                                if verbose == 2:
                                    stderr.write(verb + str(round(result, 4))
                                                 + '\n')
                                else:
                                    print verb + str(round(result, 4))
                            continue
                        tmp = {'kforce'   : 5,
                               'lowrdist' : 100,
                               'maxdist'  : int(maxdist),
                               'upfreq'   : float(upfreq),
                               'lowfreq'  : float(lowfreq),
                               'scale'    : float(scale)}
                        try:
                            count += 1
                            models, bad_models = generate_3d_models(
                                self.zscores, self.resolution,
                                self.nloci, n_models=self.n_models,
                                n_keep=self.n_keep, config=tmp,
                                n_cpus=n_cpus, first=0,
                                values=self.values, container=self.container,
                                close_bins=self.close_bins, zeros=self.zeros)
                            self.__models = models
                            self._bad_models = bad_models
                            result = 0
                            cutoff = my_round(dcutoff_arange[0])
                            for cut in [i for i in dcutoff_arange]:
                                sub_result = self.correlate_with_real_data(
                                    cutoff=(int(cut * self.resolution *
                                                float(scale))),
                                    off_diag=off_diag)[0]
                                if result < sub_result:
                                    result = sub_result
                                    cutoff = my_round(cut)
                        except Exception, e:
                            print '  SKIPPING: %s' % e
                            result = 0
                            cutoff = my_round(dcutoff_arange[0])
                        if verbose:
                            verb = '%5s %6s %7s %7s %6s %7s  ' % (
                                count, upfreq, lowfreq, maxdist,
                                scale, cutoff)
                            if verbose == 2:
                                stderr.write(verb + str(round(result, 4))
                                             + '\n')
                            else:
                                print verb + str(round(result, 4))
                        # store
                        self.results[(scale, maxdist,
                                      upfreq, lowfreq, cutoff)] = result
                        
        self.scale_range.sort(  key=float)
        self.maxdist_range.sort(key=float)
        self.lowfreq_range.sort(key=float)
        self.upfreq_range.sort( key=float)
        self.dcutoff_range.sort(key=float)


    def get_best_parameters_dict(self, reference=None, with_corr=False):
        """
        :param None reference: a description of the dataset optimized
        :param False with_corr: if True, returns also the correlation value

        :returns: a dict that can be used for modelling, see config parameter in
           :func:`pytadbit.experiment.Experiment.model_region`
           
        """
        if not self.results:
            stderr.write('WARNING: no optimization done yet\n')
            return
        best = ((None, None, None, None), 0.0)
        for (sca, mxd, ufq, lfq, cut), val in self.results.iteritems():
            if val > best[-1]:
                best = ((sca, mxd, ufq, lfq, cut), val)
        if with_corr:
            return (dict((('scale'  , float(best[0][0])),
                          ('maxdist', float(best[0][1])),
                          ('upfreq' , float(best[0][2])),
                          ('lowfreq', float(best[0][3])),
                          ('dcutoff', float(best[0][4])),
                          ('reference', reference or ''), ('kforce', 5))),
                    best[-1])
        else:
            return dict((('scale'  , float(best[0][0])),
                         ('maxdist', float(best[0][1])),
                         ('upfreq' , float(best[0][2])),
                         ('lowfreq', float(best[0][3])),
                         ('dcutoff', float(best[0][4])),
                         ('reference', reference or ''), ('kforce', 5)))
    


    def _result_to_array(self):
        results = np.empty((len(self.scale_range), len(self.maxdist_range),
                            len(self.upfreq_range), len(self.lowfreq_range)))
        for w, scale in enumerate(self.scale_range):
            for x, maxdist in enumerate(self.maxdist_range):
                for y, upfreq in enumerate(self.upfreq_range):
                    for z, lowfreq in enumerate(self.lowfreq_range):
                        try:
                            cut = [c for c in self.dcutoff_range
                                   if (scale, maxdist, upfreq, lowfreq, c)
                                   in self.results][0]
                        except IndexError:
                            results[w, x, y, z] = float('nan')
                            continue
                        try:
                            results[w, x, y, z] = self.results[
                                (scale, maxdist, upfreq, lowfreq, cut)]
                        except KeyError:
                            results[w, x, y, z] = float('nan')
        return results


    def write_result(self, f_name):
        """
        This function writes a log file of all the values tested for each 
        parameter, and the resulting correlation value.

        This file can be used to load or merge data a posteriori using 
        the function pytadbit.imp.impoptimizer.IMPoptimizer.load_from_file
        
        :param f_name: file name with the absolute path
        """
        out = open(f_name, 'w')
        out.write(('## n_models: %s n_keep: %s ' +
                   'close_bins: %s\n') % (self.n_models, 
                                          self.n_keep, self.close_bins))
        out.write('# scale\tmax_dist\tup_freq\tlow_freq\tdcutoff\tcorrelation\n')
        for scale in self.scale_range:
            for maxdist in self.maxdist_range:
                for upfreq in self.upfreq_range:
                    for lowfreq in self.lowfreq_range:
                        try:
                            cut = sorted(
                                [c for c in self.dcutoff_range
                                 if (scale, maxdist, upfreq, lowfreq, c)
                                 in self.results],
                                key=lambda x: self.results[
                                    (scale, maxdist, upfreq, lowfreq, x)])[0]
                        except IndexError:
                            print 'Missing dcutoff', (scale, maxdist, upfreq, lowfreq)
                            continue
                        try:
                            result = self.results[(scale, maxdist,
                                                   upfreq, lowfreq, cut)]
                            out.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
                                scale, maxdist, upfreq, lowfreq, cut, result))
                        except KeyError:
                            print 'KeyError', (scale, maxdist, upfreq, lowfreq, cut, result)
                            continue
        out.close()


    def load_from_file(self, f_name):
        """
        Loads the optimized parameters from a file generated with the function:
        pytadbit.imp.impoptimizer.IMPoptimizer.write_result.
        This function does not overwrite the parameters that were already 
        loaded or calculated.

        :param f_name: file name with the absolute path
        """
        for line in open(f_name):
            # Check same parameters
            if line.startswith('##'):
                n_models, _, n_keep, _, close_bins = line.split()[2:]
                if ([int(n_models), int(n_keep), int(close_bins)]
                    != 
                    [self.n_models, self.n_keep, self.close_bins]):
                    raise Exception('Parameters does in %s not match: %s\n%s' %(
                        f_name,
                        [int(n_models), int(n_keep), int(close_bins)],
                        [self.n_models, self.n_keep, self.close_bins]))
            if line.startswith('#'):
                continue
            scale, maxdist, upfreq, lowfreq, dcutoff, result = line.split()
            scale, maxdist, upfreq, lowfreq, dcutoff = (
                float(scale), int(maxdist), float(upfreq), float(lowfreq),
                float(dcutoff))
            scale   = my_round(scale, val=5)
            maxdist = my_round(maxdist)
            upfreq  = my_round(upfreq)
            lowfreq = my_round(lowfreq)
            dcutoff = my_round(dcutoff)
            self.results[(scale, maxdist, upfreq, lowfreq, dcutoff)] = float(result)
            if not scale in self.scale_range:
                self.scale_range.append(scale)
            if not maxdist in self.maxdist_range:
                self.maxdist_range.append(maxdist)
            if not upfreq in self.upfreq_range:
                self.upfreq_range.append(upfreq)
            if not lowfreq in self.lowfreq_range:
                self.lowfreq_range.append(lowfreq)
            if not dcutoff in self.dcutoff_range:
                self.dcutoff_range.append(dcutoff)
        self.scale_range.sort(  key=float)
        self.maxdist_range.sort(key=float)
        self.lowfreq_range.sort(key=float)
        self.upfreq_range.sort( key=float)
        self.dcutoff_range.sort(key=float)
    
    
                
    def correlate_with_real_data(self, cutoff=None, off_diag=1):
        
        if not cutoff:
            cutoff = int(2 * self.resolution * self._config['scale'])
        model_matrix = self.get_contact_matrix(cutoff=cutoff)
        oridata = []
        moddata = []
        for i in xrange(len(self.norm)):
            for j in xrange(i + off_diag, len(self.norm)):
                if not self.norm[i][j] > 0:
                    continue
                oridata.append(self.norm[i][j])
                moddata.append(model_matrix[i][j])
        corr = spearmanr(moddata, oridata)
        
        return corr
    
    def get_contact_matrix(self, cutoff=None):
        
        models = [m for m in self.__models]
        
        matrix = [[float('nan') for _ in xrange(self.nloci)]
                  for _ in xrange(self.nloci)]
        if not cutoff:
            cutoff = int(2 * self.resolution * self._config['scale'])
        cutoff = cutoff**2
        for i in xrange(self.nloci):
            for j in xrange(i + 1, self.nloci):
                val = len([k for k in self.__square_3d_dist(
                    i + 1, j + 1)
                           if k < cutoff])
                matrix[i][j] = matrix[j][i] = float(val) / len(models)  # * 100
        return matrix
    
    def __square_3d_dist(self, part1, part2):
        """
        same as median_3d_dist, but return the square of the distance instead
        """
        part1 -= 1
        part2 -= 1
        #models = [m for m in self.__models]
        return [(self.__models[mdl]['x'][part1] - self.__models[mdl]['x'][part2])**2 +
                (self.__models[mdl]['y'][part1] - self.__models[mdl]['y'][part2])**2 +
                (self.__models[mdl]['z'][part1] - self.__models[mdl]['z'][part2])**2
                for mdl in self.__models]
    
def my_round(num, val=4):
    num = round(float(num), val)
    return str(int(num) if num == int(num) else num)


def zscore(values):
    """
    Calculates the log10, Z-score of a given list of values.
    
    .. note::
    
      _______________________/___
                            /
                           /
                          /
                         /
                        /
                       /
                      /
                     /
                    /
                   /
                  /
                 /
                /                     score
            ___/_________________________________
              /
  
    """
    # get the log trasnform values
    nozero_log(values)
    mean_v = np.mean(values.values())
    std_v  = np.std (values.values())
    # replace values by z-score
    for i in values:
        values[i] = (values[i] - mean_v) / std_v

def transform(val):
    return log10(val)

def nozero_log(values):
    # Set the virtual minimum of the matrix to half the non-null real minimum
    minv = float(min([v for v in values.values() if v])) / 2
    # if minv > 1:
    #     warn('WARNING: probable problem with normalization, check.\n')
    #     minv /= 2  # TODO: something better
    logminv = transform(minv)
    for i in values:
        try:
            values[i] = transform(values[i])
        except ValueError:
            values[i] = logminv

def get_hic_zscores(norm, size):
        
    values = {}
    zeros  = {}
    zscores = {}
    
    for i in xrange(size):
        if not norm[i][i]:
            zeros[i] = i 
    
    for i in xrange(size):
        if i in zeros:
            continue
        for j in xrange(i + 1, size):
            if j in zeros:
                continue
            if (not norm[i][j]):
                zeros[(i, j)] = None
                continue
            values[(i, j)] = norm[i][j]
                
    # compute Z-score
    zscore(values)
    for i in xrange(size):
        if i in zeros:
            continue
        for j in xrange(i + 1, size):
            if j in zeros:
                continue
            if (i, j) in zeros:
                continue
            zscores.setdefault(str(i), {})
            zscores[str(i)][str(j)] = values[(i, j)]
    
    return zscores, values, zeros
                
def generate_3d_models(zscores, resolution, nloci, start=1, n_models=5000,
                       n_keep=1000, close_bins=1, n_cpus=1, keep_all=False,
                       verbose=0, outfile=None, config=None,
                       values=None, experiment=None, coords=None, zeros=None,
                       first=None, container=None):
        """
        This function generates three-dimensional models starting from Hi-C data. 
        The final analysis will be performed on the n_keep top models.
        
        :param zscores: the dictionary of the Z-score values calculated from the 
           Hi-C pairwise interactions
        :param resolution:  number of nucleotides per Hi-C bin. This will be the 
           number of nucleotides in each model's particle
        :param nloci: number of particles to model (may not all be present in
           zscores)
        :param None experiment: experiment from which to do the modelling (used only
           for descriptive purpose)
        :param None coords: a dictionary like:
           ::
    
             {'crm'  : '19',
              'start': 14637,
              'end'  : 15689}
    
        :param 5000 n_models: number of models to generate
        :param 1000 n_keep: number of models used in the final analysis (usually 
           the top 20% of the generated models). The models are ranked according to
           their objective function value (the lower the better)
        :param False keep_all: whether or not to keep the discarded models (if 
           True, models will be stored under StructuralModels.bad_models) 
        :param 1 close_bins: number of particles away (i.e. the bin number 
           difference) a particle pair must be in order to be considered as
           neighbors (e.g. 1 means consecutive particles)
        :param n_cpus: number of CPUs to use
        :param False verbose: if set to True, information about the distance, force
           and Z-score between particles will be printed. If verbose is 0.5 than
           constraints will be printed only for the first model calculated.
        :param None values: the normalized Hi-C data in a list of lists (equivalent 
           to a square matrix)
        :param None config: a dictionary containing the standard 
           parameters used to generate the models. The dictionary should contain
           the keys kforce, lowrdist, maxdist, upfreq and lowfreq. Examples can be
           seen by doing:
    
           ::
    
             from pytadbit.imp.CONFIG import CONFIG
    
             where CONFIG is a dictionary of dictionaries to be passed to this function:
    
           ::
    
             CONFIG = {
              'dmel_01': {
                  # Paramaters for the Hi-C dataset from:
                  'reference' : 'victor corces dataset 2013',
    
                  # Force applied to the restraints inferred to neighbor particles
                  'kforce'    : 5,
    
                  # Maximum experimental contact distance
                  'maxdist'   : 600, # OPTIMIZATION: 500-1200
    
                  # Maximum threshold used to decide which experimental values have to be
                  # included in the computation of restraints. Z-score values greater than upfreq
                  # and less than lowfreq will be included, while all the others will be rejected
                  'upfreq'    : 0.3, # OPTIMIZATION: min/max Z-score
    
                  # Minimum thresholds used to decide which experimental values have to be
                  # included in the computation of restraints. Z-score values bigger than upfreq
                  # and less that lowfreq will be include, whereas all the others will be rejected
                  'lowfreq'   : -0.7 # OPTIMIZATION: min/max Z-score
    
                  # Space occupied by a nucleotide (nm)
                  'scale'     : 0.005
    
                  }
              }
        :param None first: particle number at which model should start (0 should be
           used inside TADbit)
        :param None container: restrains particle to be within a given object. Can 
           only be a 'cylinder', which is, in fact a cylinder of a given height to 
           which are added hemispherical ends. This cylinder is defined by a radius, 
           its height (with a height of 0 the cylinder becomes a sphere) and the 
           force applied to the restraint. E.g. for modeling E. coli genome (2 
           micrometers length and 0.5 micrometer of width), these values could be 
           used: ['cylinder', 250, 1500, 50], and for a typical mammalian nuclei
           (6 micrometers diameter): ['cylinder', 3000, 0, 50]
    
        :returns: a StructuralModels object
    
        """
    
        # Main config parameters
        global CONFIG
        CONFIG = config or CONFIG['dmel_01']
        CONFIG['kforce'] = CONFIG.get('kforce', 5)
    
        # setup container
        try:
            CONFIG['container'] = {'shape' : container[0],
                                   'radius': container[1],
                                   'height': container[2],
                                   'cforce': container[3]}
        except:
            CONFIG['container'] = {'shape' : None,
                                   'radius': None,
                                   'height': None,
                                   'cforce': None}
        # Particles initial radius
        global RADIUS
        RADIUS = float(resolution * CONFIG['scale']) / 2
        CONFIG['lowrdist'] = RADIUS * 2.
        
    
        if CONFIG['lowrdist'] > CONFIG['maxdist']:
            raise TADbitModelingOutOfBound(
                ('ERROR: we must prevent you from doing this for the safe of our' +
                 'universe...\nIn this case, maxdist must be higher than %s\n' +
                 '   -> resolution times scale -- %s*%s)') % (
                    CONFIG['lowrdist'], resolution, CONFIG['scale']))
    
        # get SLOPE and regression for all particles of the z-score data
        global SLOPE, INTERCEPT
        zsc_vals = [zscores[i][j] for i in zscores for j in zscores[i]
                    if abs(int(i) - int(j)) > 1] # condition is to avoid
                                                 # taking into account selfies
                                                 # and neighbors
        SLOPE, INTERCEPT   = polyfit([min(zsc_vals), max(zsc_vals)],
                                     [CONFIG['maxdist'], CONFIG['lowrdist']], 1)
        # get SLOPE and regression for neighbors of the z-score data
        global NSLOPE, NINTERCEPT
        xarray = [zscores[i][j] for i in zscores for j in zscores[i]
                  if abs(int(i) - int(j)) <= (close_bins + 1)]
        yarray = [RADIUS * 2 for _ in xrange(len(xarray))]
        NSLOPE, NINTERCEPT = polyfit(xarray, yarray, 1)
        
        global LOCI
        # if z-scores are generated outside TADbit they may not start at zero
        if first == None:
            first = min([int(j) for i in zscores for j in zscores[i]] +
                        [int(i) for i in zscores])
        LOCI  = range(first, nloci + first)
        
        # Z-scores
        global PDIST
        PDIST = zscores
        # random inital number
        global START
        START = start
        # verbose
        global VERBOSE
        VERBOSE = verbose
        #VERBOSE = 3
    
        models, bad_models = multi_process_model_generation(
            n_cpus, n_models, n_keep, keep_all)
        if coords:
            description = {'chromosome'        : coords['crm'],
                           'start'             : resolution * coords['start'],
                           'end'               : resolution * coords['end'],
                           'resolution'        : resolution}
        for i, m in enumerate(models.values() + bad_models.values()):
            m['index'] = i
            if coords:
                m['description'] = description
        if outfile:
            if exists(outfile):
                old_models, old_bad_models = load(open(outfile))
            else:
                old_models, old_bad_models = {}, {}
            models.update(old_models)
            bad_models.update(old_bad_models)
            out = open(outfile, 'w')
            dump((models, bad_models), out)
            out.close()
        else:
            return models, bad_models

def _get_restraints():
    """
    Same function as addAllHarmonic but just to get restraints
    """
    model = {'rk'    : IMP.FloatKey("radius"),
             'model' : Model(),
             'rs'    : None, # 2.6.1 compat 
             'ps'    : None}
    model['ps'] = ListSingletonContainer(model['model'],
        IMP.core.create_xyzr_particles(model['model'], len(LOCI),
                                       RADIUS, 100000))
    model['ps'].set_name("")

    # set container
    try:
        model['rs'] = IMP.RestraintSet(model['model']) # 2.6.1 compat 
    except:
        pass
    model['container'] = CONFIG['container']

    # elif model['container']['shape']:
    #     raise noti
    
    for i in range(0, len(LOCI)):
        p = model['ps'].get_particle(i)
        p.set_name(str(LOCI[i]))
        #p.set_value(model['rk'], RADIUS)
    restraints = {}
    for i in range(len(LOCI)):
        p1 = model['ps'].get_particle(i)
        x = p1.get_name()
        if model['container']['shape'] == 'sphere':
            ub  = IMP.core.HarmonicUpperBound(
                model['container']['properties'][0], CONFIG['kforce'] * 10)
            ss  = IMP.core.DistanceToSingletonScore(
                ub, model['container']['center'])
            rss = IMP.core.SingletonRestraint(ss, p1)
            try:
                model['model'].add_restraint(rss) 
            except:
                model['rs'].add_restraint(rss) # 2.6.1 compat
            rss.evaluate(False)
        for j in range(i+1, len(LOCI)):
            p2 = model['ps'].get_particle(j)
            y = p2.get_name()
            typ, dist, frc = addHarmonicPair(model, p1, p2, x, y, j, dry=True)
            if VERBOSE >= 1:
                stdout.write('%s\t%s\t%s\t%s\t%s\n' % (typ, x, y, dist, frc))
            if typ[-1] == 'a':
                typ = 'H'
            elif typ[-1] == 'l':
                typ = 'L'
            elif typ[-1] == 'u':
                typ = 'U'
            elif typ[-1] == 'n':
                typ = 'C'
            else:
                continue
            restraints[tuple(sorted((x, y)))] = typ[-1], dist, frc
    return restraints

def multi_process_model_generation(n_cpus, n_models, n_keep, keep_all):
    """
    Parallelize the
    :func:`pytadbit.imp.imp_model.StructuralModels.generate_IMPmodel`.

    :param n_cpus: number of CPUs to use
    :param n_models: number of models to generate
    """

    pool = mu.Pool(n_cpus)
    jobs = {}
    for rand_init in xrange(START, n_models + START):
        jobs[rand_init] = pool.apply_async(generate_IMPmodel,
                                           args=(rand_init,))

    pool.close()
    pool.join()

    results = []
    for rand_init in xrange(START, n_models + START):
        results.append((rand_init, jobs[rand_init].get()))   

    models = {}
    bad_models = {}
    for i, (_, m) in enumerate(
        sorted(results, key=lambda x: x[1]['objfun'])[:n_keep]):
        models[i] = m
    if keep_all:
        for i, (_, m) in enumerate(
        sorted(results, key=lambda x: x[1]['objfun'])[n_keep:]):
            bad_models[i+n_keep] = m
    return models, bad_models

def generate_IMPmodel(rand_init):
    """
    Generates one IMP model
    
    :param rand_init: random number kept as model key, for reproducibility.

    :returns: a model, that is a dictionary with the log of the objective
       function value optimization, and the coordinates of each particles.

    """
    verbose = VERBOSE
    IMP.random_number_generator.seed(rand_init)

    log_energies = []
    model = {'rk'    : IMP.FloatKey("radius"),
             'model' : Model(),
             'rs'    : None, # 2.6.1 compat
             'ps'    : None,
             'pps'   : None}
    model['ps'] = ListSingletonContainer(model['model'],
        IMP.core.create_xyzr_particles(model['model'], len(LOCI),
                                       RADIUS, 100000))
    model['ps'].set_name("")

    # initialize each particles
    for i in range(0, len(LOCI)):
        p = model['ps'].get_particle(i)
        p.set_name(str(LOCI[i]))
        # computed following the relationship with the 30nm vs 40nm fiber
        #p.set_value(model['rk'], RADIUS)

    # Restraints between pairs of LOCI proportional to the PDIST
    try:
        model['pps']  = IMP.kernel.ParticlePairsTemp()
    except:
        model['pps']  = IMP.ParticlePairsTemp() # 2.6.1 compat

    # CALL BIG FUNCTION
    if rand_init == START and verbose == 0.5:
        verbose = 1
        stdout.write("# Harmonic\tpart1\tpart2\tdist\tkforce\n")

    # set container
    try:
        model['rs'] = IMP.RestraintSet(model['model']) # 2.6.1 compat
    except:
        pass
    restraints = [] # 2.6.1 compat
    model['container'] = CONFIG['container']    
    # elif model['container']['shape']:
    #     raise noti

    addAllHarmonics(model)

    # Setup an excluded volume restraint between a bunch of particles
    # with radius
    r = IMP.core.ExcludedVolumeRestraint(model['ps'], CONFIG['kforce'])
    try:
        model['model'].add_restraint(r)
    except:
        model['rs'].add_restraint(r) # 2.6.1 compat
        restraints.append(model['rs'])
        scoring_function = IMP.core.RestraintsScoringFunction(restraints)

    if verbose == 3:
       try:
           "Total number of restraints: %i" % (
              model['model'].get_number_of_restraints())
       except:
           "Total number of restraints: %i" % (
              model['rs'].get_number_of_restraints()) # 2.6.1 compat

    # Set up optimizer
    try:
        lo = IMP.core.ConjugateGradients()
        lo.set_model(model['model'])
    except: # since version 2.5, IMP goes this way
        lo = IMP.core.ConjugateGradients(model['model'])
    try:
        lo.set_scoring_function(scoring_function) # 2.6.1 compat
    except:
        pass
    o = IMP.core.MonteCarloWithLocalOptimization(lo, LSTEPS)
    try:
        o.set_scoring_function(scoring_function) # 2.6.1 compat
    except:
        pass
    o.set_return_best(True)
    fk = IMP.core.XYZ.get_xyz_keys()
    ptmp = model['ps'].get_particles()
    mov = IMP.core.NormalMover(ptmp, fk, 0.25)
    o.add_mover(mov)
    # o.add_optimizer_state(log)

    # Optimizer's parameters
    if verbose == 3:
         "nrounds: %i, steps: %i, lsteps: %i" % (NROUNDS, STEPS, LSTEPS)

    # Start optimization and save an VRML after 100 MC moves
    try:
         log_energies.append(model['model'].evaluate(False))
    except:
         log_energies.append(model['rs'].evaluate(False)) # 2.6.1 compat
    if verbose == 3:
         "Start", log_energies[-1]

    #"""simulated_annealing: preform simulated annealing for at most nrounds
    # iterations. The optimization stops if the score does not change more than
    #    a value defined by endLoopValue and for stopCount iterations. 
    #   @param endLoopCount = Counter that increments if the score of two models
    # did not change more than a value
    #   @param stopCount = Maximum values of iteration during which the score
    # did not change more than a specific value
    #   @paramendLoopValue = Threshold used to compute the value  that defines
    # if the endLoopCounter should be incremented or not"""
    # IMP.fivec.simulatedannealing.partial_rounds(m, o, nrounds, steps)
    endLoopCount = 0
    stopCount = 10
    endLoopValue = 0.00001
    # alpha is a parameter that takes into account the number of particles in
    # the model (len(LOCI)).
    # The multiplier (in this case is 1.0) is used to give a different weight
    # to the number of particles
    alpha = 1.0 * len(LOCI)
    # During the firsts hightemp iterations, do not stop the optimization
    hightemp = int(0.025 * NROUNDS)
    for i in range(0, hightemp):
        temperature = alpha * (1.1 * NROUNDS - i) / NROUNDS
        o.set_kt(temperature)
        log_energies.append(o.optimize(STEPS))
        if verbose == 3:
             i, log_energies[-1], o.get_kt()
    # After the firsts hightemp iterations, stop the optimization if the score
    # does not change by more than a value defined by endLoopValue and
    # for stopCount iterations
    lownrj = log_energies[-1]
    for i in range(hightemp, NROUNDS):
        temperature = alpha * (1.1 * NROUNDS - i) / NROUNDS
        o.set_kt(temperature)
        log_energies.append(o.optimize(STEPS))
        if verbose == 3:
            print i, log_energies[-1], o.get_kt()
        # Calculate the score variation and check if the optimization
        # can be stopped or not
        if lownrj > 0:
            deltaE = fabs((log_energies[-1] - lownrj) / lownrj)
        else:
            deltaE = log_energies[-1]
        if (deltaE < endLoopValue and endLoopCount == stopCount):
            break
        elif (deltaE < endLoopValue and endLoopCount < stopCount):
            endLoopCount += 1
            lownrj = log_energies[-1]
        else:
            endLoopCount = 0
            lownrj = log_energies[-1]
    #"""simulated_annealing_full: preform simulated annealing for nrounds
    # iterations"""
    # # IMP.fivec.simulatedannealing.full_rounds(m, o, nrounds, steps)
    # alpha = 1.0 * len(LOCI)
    # for i in range(0,nrounds):
    #    temperature = alpha * (1.1 * nrounds - i) / nrounds
    #    o.set_kt(temperature)
    #    e = o.optimize(steps)
    #    print str(i) + " " + str(e) + " " + str(o.get_kt())

    try:
        log_energies.append(model['model'].evaluate(False))
    except:
        log_energies.append(model['rs'].evaluate(False)) # 2.6.1 compat
    if verbose >=1:
        if verbose >= 2 or not rand_init % 100:
            print 'Model %s IMP Objective Function: %s' % (
                rand_init, log_energies[-1])
    x, y, z, radius = (FloatKey("x"), FloatKey("y"),
                       FloatKey("z"), FloatKey("radius"))
    result = IMPmodel({'log_objfun' : log_energies,
                       'objfun'     : log_energies[-1],
                       'x'          : [],
                       'y'          : [],
                       'z'          : [],
                       'radius'     : None,
                       'cluster'    : 'Singleton',
                       'rand_init'  : str(rand_init)})
    for part in model['ps'].get_particles():
        result['x'].append(part.get_value(x))
        result['y'].append(part.get_value(y))
        result['z'].append(part.get_value(z))
        if verbose == 3:
            print (part.get_name(), part.get_value(x), part.get_value(y),
                   part.get_value(z), part.get_value(radius))
    # gets radius from last particle, assuming that all are the same
    result['radius'] = part.get_value(radius)
    return result # rand_init, result

def addAllHarmonics(model):
    """
    Add harmonics to all pair of particles.
    """
    for i in range(len(LOCI)):
        p1 = model['ps'].get_particle(i)
        x = p1.get_name()
        for j in range(i+1, len(LOCI)):
            p2 = model['ps'].get_particle(j)
            y = p2.get_name()
            addHarmonicPair(model, p1, p2, x, y, j)


def addHarmonicPair(model, p1, p2, x, y, j, dry=False):
    """
    add harmonic to a given pair of particles
    :param model: a model dictionary that contains IMP model, singleton
       containers...
    :param p1: first particle
    :param p2: second particle
    :param x: first particle name
    :param y: second particle name
    :param j: id of second particle
    :param num_loci1: index of the first particle
    :param num_loci2: index of the second particle
    """
    num_loci1, num_loci2 = int(x), int(y)
    seqdist = num_loci2 - num_loci1
    restraint = ('no', 0, 0)
    freq = float('nan')
    # SHORT RANGE DISTANCE BETWEEN TWO CONSECUTIVE LOCI
    if seqdist == 1:
        kforce = CONFIG['kforce']
        if x in PDIST and y in PDIST[x] and PDIST[x][y] > CONFIG['upfreq']:
            dist = distConseq12(PDIST[p1.get_name()][p2.get_name()])
            if not dry:
                addHarmonicNeighborsRestraints(model, p1, p2, dist, kforce)
            else:
                return ("addHn", dist, kforce)
        else:
            dist = (p1.get_value(model['rk']) + p2.get_value(model['rk']))
            # dist = (p1.get_value(rk) + p2.get_value(rk))
            if not dry:
                addHarmonicUpperBoundRestraints(model, p1, p2, dist, kforce)
            else:
                return ("addHu", dist, kforce)
    # SHORT RANGE DISTANCE BETWEEN TWO SEQDIST = 2
    elif seqdist == 2:
        p3 = model['ps'].get_particle(j-1)
        kforce = CONFIG['kforce']
        dist = (p1.get_value(model['rk']) + p2.get_value(model['rk'])
                + 2.0 * p3.get_value(model['rk']))
        # dist = (p1.get_value(rk) + p2.get_value(rk))
        if not dry:
            addHarmonicUpperBoundRestraints(model, p1, p2, dist, kforce)
        else:
            return ("addHu", dist, kforce)
    # LONG RANGE DISTANCE DISTANCE BETWEEN TWO NON-CONSECUTIVE LOCI
    elif x in PDIST and y in PDIST[x]:
        freq = PDIST[x][y]
        kforce = kForce(freq)
    # X IN PDIST BUT Y NOT IN PDIST[X]
    elif x in PDIST:
        prevy = str(num_loci2 - 1)
        posty = str(num_loci2 + 1)
        # mean dist to prev and next part are used with half weight
        freq = (PDIST[x].get(prevy, PDIST[x].get(posty, float('nan'))) +
                PDIST[x].get(posty, PDIST[x].get(prevy, float('nan')))) / 2

        kforce = 0.5 * kForce(freq)
    # X NOT IN PDIST
    else:
        prevx = str(num_loci1 - 1)
        postx = str(num_loci1 + 1)
        prevx = prevx if prevx in PDIST else postx
        postx = postx if postx in PDIST else prevx
        try:
            freq = (PDIST[prevx].get(y, PDIST[postx].get(y, float('nan'))) +
                    PDIST[postx].get(y, PDIST[prevx].get(y, float('nan')))) / 2
        except KeyError:
            pass
        kforce = 0.5 * kForce(freq)

    # FREQUENCY > UPFREQ
    if freq > CONFIG['upfreq']:
        if not dry:
            addHarmonicRestraints(model, p1, p2, distance(freq), kforce)
        else:
            return ("addHa", distance(freq), kforce)
    # FREQUENCY > LOW THIS HAS TO BE THE THRESHOLD FOR
    # "PHYSICAL INTERACTIONS"
    elif freq < CONFIG['lowfreq']:
        if not dry:
            addHarmonicLowerBoundRestraints(model, p1, p2, distance(freq), kforce)
        else:
            return ("addHl", distance(freq), kforce)
    if dry:
        return restraint

def distConseq12(freq):
    """
    Function mapping the Z-scores into distances for neighbor fragments
    """
    return (NSLOPE * freq) + NINTERCEPT

def distance(freq):
    """
    Function mapping the Z-scores into distances for non-neighbor fragments
    """
    return (SLOPE * freq) + INTERCEPT

def addHarmonicNeighborsRestraints(model, p1, p2, dist, kforce):
    p = IMP.ParticlePair(p1, p2)
    model['pps'].append(p)
    try:
        dr = IMP.core.DistanceRestraint(
            model['model'], IMP.core.Harmonic(dist, kforce),p1, p2)
    except TypeError:
        dr = IMP.core.DistanceRestraint(
            IMP.core.Harmonic(dist, kforce),p1, p2) # older versions
    try:
        model['model'].add_restraint(dr)
    except:
        model['rs'].add_restraint(dr) # 2.6.1 compat

def addHarmonicUpperBoundRestraints(model, p1, p2, dist, kforce):
    p = IMP.ParticlePair(p1, p2)
    model['pps'].append(p)
    try:
        dr = IMP.core.DistanceRestraint(
            model['model'], IMP.core.HarmonicUpperBound(dist, kforce), p1, p2)
    except TypeError:
        dr = IMP.core.DistanceRestraint(
            IMP.core.HarmonicUpperBound(dist, kforce), p1, p2) # older versions
    try:
        model['model'].add_restraint(dr)
    except:
        model['rs'].add_restraint(dr) # 2.6.1 compat

def addHarmonicRestraints(model, p1, p2, dist, kforce):
    p = IMP.ParticlePair(p1, p2)
    model['pps'].append(p)
    try:
        dr = IMP.core.DistanceRestraint(
            model['model'], IMP.core.Harmonic(dist, kforce), p1, p2)
    except TypeError:
        dr = IMP.core.DistanceRestraint(
            IMP.core.Harmonic(dist, kforce), p1, p2) # older versions
    try:
        model['model'].add_restraint(dr)
    except:
        model['rs'].add_restraint(dr) # 2.6.1 compat

def addHarmonicLowerBoundRestraints(model, p1, p2, dist, kforce):
    p = IMP.ParticlePair(p1, p2)
    model['pps'].append(p)
    try:
        dr = IMP.core.DistanceRestraint(
            model['model'], IMP.core.HarmonicLowerBound(dist, kforce), p1, p2)
    except TypeError:
        dr = IMP.core.DistanceRestraint(
            IMP.core.HarmonicLowerBound(dist, kforce), p1, p2) # older versions
    try:
        model['model'].add_restraint(dr)
    except:
        model['rs'].add_restraint(dr) # 2.6.1 compat


def kForce(freq):
    """
    Function to assign to each restraint a force proportional to the underlying
    experimental value.
    """
    return power(fabs(freq), 0.5 )

class IMPmodel(dict):
    """
    A container for the IMP modeling results. The container is a dictionary
    with the following keys:

    - log_objfun: The list of IMP objective function values
    - objfun: The final objective function value of the corresponding model
    - rand_init: Random number generator feed (needed for model reproducibility)
    - x, y, z: 3D coordinates of each particles. Each represented as a list

    """
    def __str__(self):
        try:
            return ('IMP model ranked %s (%s particles) with: \n' +
                    ' - Final objective function value: %s\n' +
                    ' - random initial value: %s\n' +
                    ' - first coordinates:\n'+
                    '        X      Y      Z\n'+
                    '  %7s%7s%7s\n'+
                    '  %7s%7s%7s\n'+
                    '  %7s%7s%7s\n') % (
                self['index'] + 1,
                len(self['x']), self['objfun'], self['rand_init'],
                int(self['x'][0]), int(self['y'][0]), int(self['z'][0]),
                int(self['x'][1]), int(self['y'][1]), int(self['z'][1]),
                int(self['x'][2]), int(self['y'][2]), int(self['z'][2]))
        except IndexError:
            return ('IMP model of %s particles with: \n' +
                    ' - Final objective function value: %s\n' +
                    ' - random initial value: %s\n' +
                    ' - first coordinates:\n'+
                    '      X    Y    Z\n'+
                    '  %5s%5s%5s\n') % (
                len(self['x']), self['objfun'], self['rand_init'],
                self['x'][0], self['y'][0], self['z'][0])
                    
class TADbitModelingOutOfBound(Exception):
    pass