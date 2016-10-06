#!/usr/bin/python

"""
This script contains the main analysis that can be done for imp using TADbit:
  * optimization of IMP parameters
  * building of models of chromatin structure

Arguments can be passed either through command line or through a configuration
file. Note: options passed through command line override the ones from the
configuration file, that can be considered as new default values.

Computation can be divided into steps in order to be parallelized:
  - optimization of IMP parameters (using --optimize_only)
     - optimization itself can be divided setting smaller ranged of maxdist,
       lowfreq or upfreq (or even just one value).
  - modeling (using the same command as before without --optimize_only,
    optimization will be skipped if already done).

Example of usage parallelizing computation:
$ python model_and_analyze.py --cfg model_and_analyze.cfg --optimize_only --maxdist 2000
$ python model_and_analyze.py --cfg model_and_analyze.cfg --optimize_only --maxdist 2500
$ python model_and_analyze.py --cfg model_and_analyze.cfg --optimize_only --maxdist 3000

The same can be run all together with this single line:
$ python model_and_analyze.py --cfg model_and_analyze.cfg 

A log file will be generated, repeating the message appearing on console, with
line-specific flags allowing to identify from which step of the computation
belongs the message.
"""

# MatPlotLib not asking for X11
import csv
import uuid
from argparse import ArgumentParser, HelpFormatter
from IMPoptimizer import IMPoptimizer, get_hic_zscores
import os, sys
import logging
from cPickle import load, dump
from random import random
from string import ascii_letters as letters
from numpy import savetxt

    
def load_hic_data(opts, xnames):
    """
    Load Hi-C data
    """
    if os.path.isfile(opts.norm[0]):
        with open(opts.norm[0], 'r') as f_data:
            reader = csv.reader(f_data, delimiter="\t")
            data = list(reader)
            for i in xrange(len(data)):
                for j in xrange(len(data[0])):
                    data[i][j] = float(data[i][j])
    else:
        return
    
    return data

def load_optimal_imp_parameters(opts, name, exp):
    """
    Load optimal IMP parameters
    """
    # If some optimizations have finished, we load log files into a single
    # IMPoptimizer object
    logging.info(("\tReading optimal parameters available in " +
                  "%s_optimal_params.tsv") % (
                     os.path.join(opts.root_path)))

    
    results = IMPoptimizer(exp, opts.res, opts.beg,
                           opts.end, n_models=opts.nmodels_opt,
                           n_keep=opts.nkeep_opt, container=opts.container)
    # load from log files
    if not opts.optimize_only:
        for fnam in os.listdir(os.path.join(opts.root_path)):
            if fnam.endswith('.tsv') and '_optimal_params' in fnam:
                results.load_from_file(os.path.join(opts.root_path, fnam))
    return results


def optimize(results, opts, name):
    """
    Optimize IMP parameters
    """
    scale   = (tuple([float(i) for i in opts.scale.split(':')  ])
               if ':' in opts.scale   else float(opts.scale)  )
    maxdist = (tuple([int(i) for i in opts.maxdist.split(':')])
               if ':' in opts.maxdist else int(opts.maxdist))
    upfreq  = (tuple([float(i) for i in opts.upfreq.split(':') ])
               if ':' in opts.upfreq  else float(opts.upfreq) )
    lowfreq = (tuple([float(i) for i in opts.lowfreq.split(':')])
               if ':' in opts.lowfreq else float(opts.lowfreq))
    dcutoff = (tuple([float(i) for i in opts.dcutoff.split(':')])
               if ':' in opts.dcutoff else float(opts.dcutoff))
    # Find optimal parameters
    logging.info("\tFinding optimal parameters for modeling " +
                 "(this can take long)...")
    optname = '_{0}_{1}_{2}_{3}_{4}'.format(opts.maxdist,
                                       opts.upfreq ,
                                       opts.lowfreq,
                                       opts.scale,
                                       opts.dcutoff)
    logpath = os.path.join(
        opts.outdir, name, '%s_optimal_params%s.tsv' % (name, optname))

    tmp_name = ''.join([letters[int(random()*52)]for _ in xrange(50)])

    tmp = open('_tmp_results_' + tmp_name, 'w')
    dump(results, tmp)
    tmp.close()
    
    tmp = open('_tmp_opts_' + tmp_name, 'w')
    dump(opts, tmp)
    tmp.close()
    
    tmp = open('_tmp_optim_' + tmp_name + '.py', 'w')
    tmp.write('''
from cPickle import load, dump

tmp_name = "%s"

results_file = open("_tmp_results_" + tmp_name)
results = load(results_file)
results_file.close()

opts_file = open("_tmp_opts_" + tmp_name)
opts = load(opts_file)
opts_file.close()

scale   = (tuple([float(i) for i in opts.scale.split(":")  ])
           if ":" in opts.scale   else float(opts.scale)  )
maxdist = (tuple([int(i) for i in opts.maxdist.split(":")])
           if ":" in opts.maxdist else int(opts.maxdist))
upfreq  = (tuple([float(i) for i in opts.upfreq.split(":") ])
           if ":" in opts.upfreq  else float(opts.upfreq) )
lowfreq = (tuple([float(i) for i in opts.lowfreq.split(":")])
           if ":" in opts.lowfreq else float(opts.lowfreq))
dcutoff = (tuple([float(i) for i in opts.dcutoff.split(":")])
           if ":" in opts.dcutoff else float(opts.dcutoff))
optname = "_{0}_{1}_{2}_{3}_{4}".format(opts.maxdist, opts.upfreq ,
                             opts.lowfreq, opts.scale,
                             opts.dcutoff)
name = "%s"
results.run_grid_search(n_cpus=opts.ncpus, off_diag=2, verbose=True,
                        lowfreq_range=lowfreq, upfreq_range=upfreq,
                        maxdist_range=maxdist, scale_range=scale,
                        dcutoff_range=dcutoff)

tmp = open("_tmp_results_" + tmp_name, "w")
dump(results, tmp)
tmp.close()
''' % (tmp_name, name))
    #results.run_grid_search(n_cpus=opts.ncpus, off_diag=2, verbose=True,
    #                       lowfreq_range=lowfreq, upfreq_range=upfreq,
    #                       maxdist_range=maxdist, scale_range=scale,
    #                       dcutoff_range=dcutoff)
    tmp.close()
    os.system("python _tmp_optim_%s.py" % tmp_name)

    results_file = open("_tmp_results_" + tmp_name)
    results = load(results_file)
    results_file.close()
    os.system('rm -f _tmp_results_%s' % (tmp_name))
    os.system('rm -f _tmp_optim_%s.py' % (tmp_name))
    os.system('rm -f _tmp_opts_%s' % (tmp_name))
    results.write_result(logpath)
    if opts.optimize_only:
        logging.info('Optimization done.')
        exit()

    ## get best parameters
    optpar, cc = results.get_best_parameters_dict(
        reference='Optimized for %s' % (name), with_corr=True)

    sc = optpar['scale']
    md = optpar['maxdist']
    uf = optpar['upfreq']
    lf = optpar['lowfreq']
    dc = optpar['dcutoff']

    logging.info(("\t\tOptimal values: scale:{0} maxdist:{1} upper:{2} " +
                  "lower:{3} dcutoff:{4} with cc: {5:.4}"
                  ).format(sc, md, uf, lf, dc, cc))
    results.write_result(os.path.join(
        opts.outdir, name, '%s_optimal_params.tsv' % (name)))

    # Optimal parameters
    kf = 5 # IMP penalty for connectivity between two consecutive particles.
           # This needs to be large enough to ensure connectivity.

    optpar['kforce'] = kf # this is already the default but it can be changed
                          # like this
    return optpar


def model_region(exp, optpar, opts, name):
    """
    generate structural models
    """
    zscores, values, zeros = get_hic_zscores(exp, len(exp[0]))

    tmp_name = ''.join([letters[int(random()*52)]for _ in xrange(50)])
    
    
    tmp = open('_tmp_zscore_' + tmp_name, 'w')
    dump([zscores, values, zeros, optpar], tmp)
    tmp.close()

    tmp = open('_tmp_opts_' + tmp_name, 'w')
    dump(opts, tmp)
    tmp.close()

    tmp = open('_tmp_model_' + tmp_name + '.py', 'w')
    tmp.write('''
from cPickle import load, dump
from IMPoptimizer import generate_3d_models
import os

tmp_name = "%s"

zscore_file = open("_tmp_zscore_" + tmp_name)
zscores, values, zeros, optpar = load(zscore_file)
zscore_file.close()

opts_file = open("_tmp_opts_" + tmp_name)
opts = load(opts_file)
opts_file.close()

nloci = opts.end - opts.beg + 1
coords = {"crm"  : opts.crm,
          "start": opts.beg,
          "end"  : opts.end}

zeros = tuple([i not in zeros for i in xrange(opts.end - opts.beg + 1)])

models, bad_models =  generate_3d_models(zscores, opts.res, nloci,
                            values=values, n_models=opts.nmodels_mod,
                            n_keep=opts.nkeep_mod,
                            n_cpus=opts.ncpus,
                            keep_all=True,
                            first=0, container=opts.container,
                            config=optpar, verbose=0.5,
                            coords=coords, zeros=zeros)
# Save models
tmp = open("_tmp_models_" + tmp_name, "w")
dump(models, tmp)
tmp.close()
''' % (tmp_name))

    tmp.close()
    os.system("python _tmp_model_%s.py" % tmp_name)
    os.system('rm -f _tmp_zscore_%s' % (tmp_name))
    os.system('rm -f _tmp_model_%s.py' % (tmp_name))
    os.system('rm -f _tmp_opts_%s' % (tmp_name))
    models_file = open("_tmp_models_" + tmp_name)
    models = load(models_file)
    models_file.close()
    
    return models


def main():
    """
    main function
    """
    opts = get_options()
    nmodels_opt, nkeep_opt, ncpus = (int(opts.nmodels_opt),
                                     int(opts.nkeep_opt), int(opts.ncpus))
    nmodels_mod, nkeep_mod = int(opts.nmodels_mod), int(opts.nkeep_mod)
    if opts.xname:
        xnames = opts.xname
    elif opts.data[0]:
        xnames = [os.path.split(d)[-1] for d in opts.data]
    else:
        xnames = [os.path.split(d)[-1] for d in opts.norm]

    name = '{0}_{1}_{2}'.format(opts.crm, opts.beg, opts.end)
    opts.outdir

    ############################################################################
    ############################  LOAD HI-C DATA  ##############################
    ############################################################################

    exp = load_hic_data(opts, xnames)


    ############################################################################
    #######################  LOAD OPTIMAL IMP PARAMETERS #######################
    ############################################################################

    results = load_optimal_imp_parameters(opts, name, exp)
        
    ############################################################################
    #########################  OPTIMIZE IMP PARAMETERS #########################
    ############################################################################

    optpar = optimize(results, opts, name)

    ############################################################################
    ##############################  MODEL REGION ###############################
    ############################################################################

    
    # Build 3D models based on the HiC data.
    logging.info("\tModeling (this can take long)...")
    models = model_region(exp, optpar, opts, name)

    contact_matrix = get_contact_matrix(models,float(opts.scale))
    
    # Save the models for easy visualization with
    # Chimera (http://www.cgl.ucsf.edu/chimera/)
    # Move into the cluster directory and run in the prompt
    # "chimera cl_1_superimpose.cmd"
    logging.info("\t\tWriting models, list and chimera files...")
    if not os.path.exists(os.path.join(opts.outdir, name, 'models')):
            os.makedirs(os.path.join(opts.outdir, name, 'models'))
    
    savetxt(fname=os.path.join(opts.outdir, name, 'models', 'contact_matrix.tsv'),X=contact_matrix, delimiter='\t')
    write_xyz(directory=os.path.join(opts.outdir, name, 'models'),models=models)
    write_cmm(directory=os.path.join(opts.outdir, name, 'models'),models=models)
    write_json(directory=os.path.join(opts.outdir, name, 'models'),models=models)
    # Write chimera file
    clschmfile = os.path.join(opts.outdir, name, 'models','superimpose.cmd')
    out = open(clschmfile, 'w')
    out.write("open " + " ".join(["model.{0}.cmm".format(models[model_n]['rand_init']) for model_n in models]))
    out.write("\nlabel; represent wire; ~bondcolor\n")
    for i in range(1, len(models) + 1):
        out.write("match #{0} #0\n".format(i-1))
    out.write("center\nscale 25")
    out.close()

def square_3d_dist(part1, part2, models):
        """
        same as median_3d_dist, but return the square of the distance instead
        """
        part1 -= 1
        part2 -= 1
        return [(models[mdl]['x'][part1] - models[mdl]['x'][part2])**2 +
                (models[mdl]['y'][part1] - models[mdl]['y'][part2])**2 +
                (models[mdl]['z'][part1] - models[mdl]['z'][part2])**2
                for mdl in models]
        
def get_contact_matrix(models, scale):
    """
    Returns a matrix with the number of interactions observed below a given
    cutoff distance.

    :param None models: if None (default) the contact matrix will be computed
       using all the models. A list of numbers corresponding to a given set
       of models can be passed
    
    :returns: matrix frequency of interaction
    """
    matrix = [[float(0.0) for _ in xrange(len(models[0]['x']))]
              for _ in xrange(len(models[0]['x']))]
    cutoff = int(2 * int(models[0]['description']['resolution']) * scale)
    cutoff = cutoff**2
    for i in xrange(len(models[0]['x'])):
        for j in xrange(i + 1, len(models[0]['x'])):
            val = len([k for k in square_3d_dist(
                i + 1, j + 1, models=models)
                       if k < cutoff])
            matrix[i][j] = matrix[j][i] = float(val) / len(models)  # * 100
    return matrix    
        
def write_cmm(directory, models, color='index'):
        """
        Save a model in the cmm format, read by Chimera
        (http://www.cgl.ucsf.edu/chimera).

        ALL the models will be written.

        :param directory: location where the file will be written (note: the
           name of the file will be model_1.cmm if model number is 1)
        :param models: list of models
        """
        for model_nbr in models:
            color_res = []
            if isinstance(color, str):
                if color == 'index':
                    for n in xrange(len(models[model_nbr]['x'])):
                        red = float(n + 1) / len(models[model_nbr]['x'])
                        color_res.append((red, 0, 1 - red))
                else:
                    raise NotImplementedError(('%s type of coloring is not yet ' +
                                               'implemeted\n') % color)
            elif not isinstance(color, list):
                raise TypeError('one of function, list or string is required\n')
            out = '<marker_set name=\"%s\">\n' % (models[model_nbr]['rand_init'])
            form = ('<marker id=\"%s\" x=\"%s\" y=\"%s\" z=\"%s\"' +
                    ' r=\"%s\" g=\"%s\" b=\"%s\" ' +
                    'radius=\"' + #str(30) +
                    str(models[model_nbr]['radius']) +
                    '\" note=\"%s\"/>\n')
            for i in xrange(len(models[model_nbr]['x'])):
                out += form % (i + 1,
                               models[model_nbr]['x'][i], models[model_nbr]['y'][i], models[model_nbr]['z'][i],
                               color_res[i][0], color_res[i][1], color_res[i][2], i + 1)
            form = ('<link id1=\"%s\" id2=\"%s\" r=\"1\" ' +
                    'g=\"1\" b=\"1\" radius=\"' + str(10) +
                    # str(self['radius']/2) +
                    '\"/>\n')
            for i in xrange(1, len(models[model_nbr]['x'])):
                out += form % (i, i + 1)
            out += '</marker_set>\n'
    
            out_f = open('%s/model.%s.cmm' % (directory,
                                                      models[model_nbr]['rand_init']), 'w')
            out_f.write(out)
            out_f.close()
        
def write_xyz(directory, models):
        """
        Writes a xyz file containing the 3D coordinates of each particle in the
        model.
        Outfile is tab separated column with the bead number being the
        first column, then the genomic coordinate and finaly the 3
        coordinates X, Y and Z

               ALL the models will be written.

        :param directory: location where the file will be written (note: the
           file name will be model.1.xyz, if the model number is 1)
        :param models: list of models
        """
    
        for model_nbr in models:
            path_f = '%s/model.%s.xyz' % (directory, models[model_nbr]['rand_init'])
            out = ''
            form = "%s\t%s\t%.3f\t%.3f\t%.3f\n"
            for i in xrange(len(models[model_nbr]['x'])):
                out += form % (
                    i + 1,
                    '%s:%s-%s' % (
                        models[model_nbr]['description']['chromosome'],
                        int(models[model_nbr]['description']['start'] or 1) + int(models[model_nbr]['description']['resolution']) * i + 1,
                        int(models[model_nbr]['description']['start'] or 1) + int(models[model_nbr]['description']['resolution']) * (i + 1)),
                    round(models[model_nbr]['x'][i], 3),
                    round(models[model_nbr]['y'][i], 3), round(models[model_nbr]['z'][i], 3))
            out_f = open(path_f, 'w')
            out_f.write(out)
            out_f.close()
        return None

def write_json(directory, models, color='index'):
    """
    Save a model in the json format, read by TADkit.

    **Note:** If none of model_num, models or cluster parameter are set,
    ALL the models will be written.

    :param directory: location where the file will be written (note: the
       name of the file will be model_1.cmm if model number is 1)
    :param models: list of models
    """
    
    color_res = []
    model_nbr = 0
    
    if isinstance(color, str):
        if color == 'index':
            for n in xrange(len(models[model_nbr]['x'])):
                red = float(n + 1) / len(models[model_nbr]['x'])
                color_res.append((red, 0, 1 - red))
        else:
            raise NotImplementedError(('%s type of coloring is not yet ' +
                                       'implemeted\n') % color)
    elif not isinstance(color, list):
        raise TypeError('one of function, list or string is required\n')
    form = '''
{
    "metadata" : {
            "version"  : 1.0,
            "type"     : "dataset",
            "generator": "TADbit"
            },
    "object": {\n%(descr)s
               "experimentType" : "Hi-C",
               "species" : "Drosophila melanogaster",
               "project" : "TADbit_paper",
               "assembly" : "DBGP 5 (dm3)",
               "identifier" : "SRR16585",
               "cellType" : "kc167",
               "uuid": "%(sha)s",
               "title": "%(title)s",
               "datatype": "xyz",
               "components": 3,
               "source": "local",
               "dependencies": ""
              },
    "models":
             [\n%(xyz)s
             ],
    "clusters":%(cluster)s,
    "centroids":%(centroid)s,
    "restraints": %(restr)s
}
'''
    fil = {}
    fil['title']   = '%s:%s-%s_%s' % (
                        models[model_nbr]['description']['chromosome'],
                        models[model_nbr]['description']['start'],
                        models[model_nbr]['description']['end'],
                        models[model_nbr]['rand_init'])

    def tocamel(key):
        key = ''.join([(w[0].upper() + w[1:]) if i else w
                       for i, w in enumerate(key.split('_'))])
        key = ''.join([(w[0].upper() + w[1:]) if i else w
                       for i, w in enumerate(key.split(' '))])
        return key
    try:
        my_descr = dict(models[model_nbr]['description'])
        my_descr['chrom'] = ["%s" % (my_descr.get('chromosome', 'Chromosome'))]
        if 'chromosome' in my_descr:
            del my_descr['chromosome']
        if 'chrom_start' not in my_descr:
            my_descr['chrom_start'] = my_descr['start']
        if 'chrom_end' not in my_descr:
            my_descr['chrom_end'] = my_descr['end']
        # coordinates inside an array in case different models
        # from different places in the genome
        my_descr['chrom_start'] = [my_descr['chrom_start']]
        my_descr['chrom_end'  ] = [my_descr['chrom_end'  ]]

        fil['descr']   = ',\n'.join([
            (' ' * 19) + '"%s" : %s' % (tocamel(k),
                                        ('"%s"' % (v))
                                        if not (isinstance(v, int) or
                                                isinstance(v, list) or
                                                isinstance(v, float))
                                        else str(v).replace("'", '"'))
            for k, v in my_descr.items()])
        if fil['descr']:
            fil['descr'] += ','
    except AttributeError:
        fil['descr']   = '"description": "Just some models"'

    fil['xyz'] = []
    for m in models:
        model = models[m]
        fil['xyz'].append((' ' * 18) + '{"ref": %s,"data": [' % (
            model['rand_init']) + ','.join(
                ['%.0f,%.0f,%.0f' % (model['x'][i],
                                     model['y'][i],
                                     model['z'][i])
                 for i in xrange(len(model['x']))]) + ']}')
    fil['xyz'] = ',\n'.join(fil['xyz'])
    fil['sha'] = str(uuid.uuid5(uuid.UUID('TADbit'.encode('hex').zfill(32)), fil['xyz']))
    fil['restr'] = '[]'
    fil['cluster'] = '[' + ','.join(['[' + models[c]['rand_init'] + ']' for c in models]) + ']'
    fil['centroid'] = '[' + ','.join([models[c]['rand_init'] for c in models]) + ']'
    path_f = '%s/models.json' % (directory)
    out_f = open(path_f, 'w')
    out_f.write(form % fil)
    out_f.close()
        
def get_options():
    """
    parse option from call

    """
    parser = ArgumentParser(
        usage="%(prog)s [options] [--cfg CONFIG_PATH]",
        formatter_class=lambda prog: HelpFormatter(prog, width=95,
                                                   max_help_position=27))
    glopts = parser.add_argument_group('General arguments')
    optimo = parser.add_argument_group('Optimization of IMP arguments')
    modelo = parser.add_argument_group('Modeling with optimal IMP arguments')
    
    parser.add_argument('--usage', dest='usage', action="store_true",
                        default=False,
                        help='''show detailed usage documentation, with examples
                        and exit''')
    parser.add_argument('--cfg', dest='cfg', metavar="PATH", action='store',
                      default=None, type=str,
                      help='path to a configuration file with predefined ' +
                      'parameters')
    parser.add_argument('--optimize_only', dest='optimize_only', default=False,
                        action='store_true',
                        help='do the optimization of the region and exit')
    parser.add_argument('--ncpus', dest='ncpus', metavar="INT", default=1,
                        type=int, help='[%(default)s] Number of CPUs to use')

    #########################################
    # GENERAL
    glopts.add_argument(
        '--root_path', dest='root_path', metavar="PATH",
        default='', type=str,
        help=('path to search for data files (just pass file name' +
              'in "data")'))
    glopts.add_argument('--data', dest='data', metavar="PATH", nargs='+',
                        type=str,
                        help='''path to file(s) with Hi-C data matrix. If many,
                        experiments will be summed up. I.e.: --data
                        replicate_1.txt replicate_2.txt''')
    glopts.add_argument('--xname', dest='xname', metavar="STR", nargs='+',
                        default=[], type=str,
                        help='''[file name] experiment name(s). Use same order
                        as data.''')
    glopts.add_argument('--norm', dest='norm', metavar="PATH", nargs='+',
                        type=str,
                        help='path to file(s) with normalizedHi-C data matrix.')
    glopts.add_argument('--crm', dest='crm', metavar="NAME",
                        help='chromosome name')
    glopts.add_argument('--beg', dest='beg', metavar="INT", type=float,
                        default=None,
                        help='genomic coordinate from which to start modeling')
    glopts.add_argument('--end', dest='end', metavar="INT", type=float,
                        help='genomic coordinate where to end modeling')
    glopts.add_argument('--res', dest='res', metavar="INT", type=int,
                        help='resolution of the Hi-C experiment')
    glopts.add_argument('--outdir', dest='outdir', metavar="PATH",
                        default=None,
                        help='out directory for results')

        #########################################
    # MODELING
    modelo.add_argument('--nmodels_mod', dest='nmodels_mod', metavar="INT",
                        default='5000', type=int,
                        help=('[%(default)s] number of models to generate for' +
                              ' modeling'))
    modelo.add_argument('--nkeep_mod', dest='nkeep_mod', metavar="INT",
                        default='1000', type=int,
                        help=('[%(default)s] number of models to keep for ' +
                        'modeling'))

    #########################################
    # OPTIMIZATION
    optimo.add_argument('--maxdist', action='store', metavar="LIST",
                        default='400', dest='maxdist',
                        help='range of numbers for maxdist' +
                        ', i.e. 400:1000:100 -- or just a number')
    optimo.add_argument('--upfreq', dest='upfreq', metavar="LIST",
                        default='0',
                        help='range of numbers for upfreq' +
                        ', i.e. 0:1.2:0.3 --  or just a number')
    optimo.add_argument('--lowfreq', dest='lowfreq', metavar="LIST",
                        default='0',
                        help='range of numbers for lowfreq' +
                        ', i.e. -1.2:0:0.3 -- or just a number')
    optimo.add_argument('--scale', dest='scale', metavar="LIST",
                        default="0.01",
                        help='[%(default)s] range of numbers to be test as ' +
                        'optimal scale value, i.e. 0.005:0.01:0.001 -- Can ' +
                        'also pass only one number')
    optimo.add_argument('--dcutoff', dest='dcutoff', metavar="LIST",
                        default="2",
                        help='[%(default)s] range of numbers to be test as ' +
                        'optimal distance cutoff parameter (distance, in ' +
                        'number of beads, from which to consider 2 beads as ' +
                        'being close), i.e. 1:5:0.5 -- Can also pass only one' +
                        ' number')
    optimo.add_argument('--nmodels_opt', dest='nmodels_opt', metavar="INT",
                        default='500', type=int,
                        help='[%(default)s] number of models to generate for ' +
                        'optimization')
    optimo.add_argument('--nkeep_opt', dest='nkeep_opt', metavar="INT",
                        default='100', type=int,
                        help='[%(default)s] number of models to keep for ' +
                        'optimization')

    parser.add_argument_group(optimo)
    parser.add_argument_group(modelo)
    opts = parser.parse_args()


    if opts.usage:
        print __doc__
        exit()

    log = '\tSummary of arguments:\n'
    # merger opts with CFG file and write summary
    args = reduce(lambda x, y: x + y, [i.strip('-').split('=')
                                       for i in sys.argv])
    new_opts = {}
    if opts.cfg:
        for line in open(opts.cfg):
            if not '=' in line:
                continue
            if line.startswith('#'):
                continue
            key, value = line.split('#')[0].strip().split('=')
            key = key.strip()
            value = value.strip()
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif key in ['data', 'norm', 'xname']:
                new_opts.setdefault(key, []).extend(value.split())
                continue
            new_opts[key] = value
    # bad key in configuration file
    for bad_k in set(new_opts.keys()) - set(opts.__dict__.keys()):
        sys.stderr.write('WARNING: parameter "%s" not recognized' % (bad_k))
    for key in sorted(opts.__dict__.keys()):
        if key in args:
            log += '  * Command setting   %13s to %s\n' % (
                key, opts.__dict__[key])
        elif key in new_opts:
            opts.__dict__[key] = new_opts[key]
            log += '  - Config. setting   %13s to %s\n' % (
                key, new_opts[key])
        else:
            log += '  o Default setting   %13s to %s\n' % (
                key, opts.__dict__[key])

    if not opts.data and not opts.norm:
        sys.stderr.write('MISSING data')
        exit(parser.print_help())
    if not opts.outdir:
        sys.stderr.write('MISSING outdir')
        exit(parser.print_help())
    if not opts.crm:
        sys.stderr.write('MISSING crm NAME')
        exit(parser.print_help())
    if not opts.beg:
        sys.stderr.write('MISSING beg COORDINATE')
        exit(parser.print_help())
    if not opts.end:
        sys.stderr.write('MISSING end COORDINATE')
        exit(parser.print_help())
    if not opts.res:
        sys.stderr.write('MISSING resolution')
        exit(parser.print_help())
    
    if not opts.maxdist:
        sys.stderr.write('MISSING maxdist')
        exit(parser.print_help())
    if not opts.lowfreq:
        sys.stderr.write('MISSING lowfreq')
        exit(parser.print_help())
    if not opts.upfreq:
        sys.stderr.write('MISSING upfreq')
        exit(parser.print_help())

    # groups for TAD detection
    if not opts.data:
        opts.data = [None] * len(opts.norm)
    else:
        opts.norm = [None] * len(opts.data)
    
    # this options should stay as this now
    # opts.scale = '0.01'

    # switch to number
    opts.nmodels_mod = int(opts.nmodels_mod)
    opts.nkeep_mod   = int(opts.nkeep_mod  )
    opts.nmodels_opt = int(opts.nmodels_opt)
    opts.nkeep_opt   = int(opts.nkeep_opt  )
    opts.ncpus       = int(opts.ncpus      )
    opts.res         = int(opts.res        )

    # TODO: UNDER TEST
    opts.container   = None #['cylinder', 1000, 5000, 100]

    # do the divisinon to bins
    opts.beg = int(float(opts.beg) / opts.res)
    opts.end = int(float(opts.end) / opts.res)
    if opts.end - opts.beg <= 2:
        raise Exception('"beg" and "end" parameter should be given in ' +
                            'genomic coordinates, not bin')

    # Create out-directory
    name = '{0}_{1}_{2}'.format(opts.crm, opts.beg, opts.end)
    if not os.path.exists(os.path.join(opts.outdir, name)):
        os.makedirs(os.path.join(opts.outdir, name))

    # write log
    if opts.optimize_only:
        log_format = '[OPTIMIZATION {0}_{1}_{2}_{3}_{4}]   %(message)s'.format(
            opts.maxdist, opts.upfreq, opts.lowfreq, opts.scale, opts.dcutoff)
    else:
        log_format = '[DEFAULT]   %(message)s'
    try:
        logging.basicConfig(filename=os.path.join(opts.outdir, name, name + '.log'),
                            level=logging.INFO, format=log_format)
    except IOError:
        logging.basicConfig(filename=os.path.join(opts.outdir, name, name + '.log2'),
                            level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(('\n' + log_format.replace('   %(message)s', '')
                  ).join(log.split('\n')))

    # update path to Hi-C data adding root directory
    if opts.root_path and opts.data[0]:
        for i in xrange(len(opts.data)):
            logging.info(os.path.join(opts.root_path, opts.data[i]))
            opts.data[i] = os.path.join(opts.root_path, opts.data[i])

    # update path to Hi-C norm adding root directory
    if opts.root_path and opts.norm[0]:
        for i in xrange(len(opts.norm)):
            logging.info(os.path.join(opts.root_path, opts.norm[i]))
            opts.norm[i] = os.path.join(opts.root_path, opts.norm[i])

    return opts


if __name__ == "__main__":
    exit(main())
