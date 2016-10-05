These scripts are extracted from [TADbit](http://sgt.cnag.cat/3dg/tadbit/), a computational framework to analyze 
and model the chromatin fiber in three dimensions, to illustrate the use of [IMP](http://integrativemodeling.org)
to model a genomic region with 5 Topologically Associating Domains (TADs) from the fly genome. 

## Directories:

- `data`:                      contains the input data and configuration files.

- `scripts`
  - `01_model_and_analyze.py`  the main IMP script modeling to model the region
  - `IMPOptimizer.py`                    contains classes and scripts to find the optimal parameters in the modelling for IMP

- `outputs` will contain the results of the script as a folder structure with:
  - .log file
  - .tsv optimal parameters' file for IMP
  - .xyz file for each model
  - .cmm [UCSF Chimera](http://www.cgl.ucsf.edu/chimera/) file for each model
  - .cmd [UCSF Chimera](http://www.cgl.ucsf.edu/chimera/) file to display all the models at the same time
  - .json file to visualize and analyze the model using [TADkit](http://sgt.cnag.cat/3dg/tadkit/). TADkit creates interactive 3D representations of chromatin conformations.

- `test` file to run the test

## Running the IMP scripts for the fly genome:
To run the modeling script, just change into its directory and run it from the
command line, e.g.
 - `cd scripts`
 - `python 01_model_and_analyze.py --cfg ../data/chr4.cfg --ncpus 12` 

A prebuild optimal imp parameters it's already included in the data directory to speed up the test which should take around 2 hours in a single cpu. 
If you wish to test the optimization and the modeling just delete the configuration file or change the extension to something else than .tsv. 
The whole computation should take around 40 hours in a single cpu.

The results will be produced in the outputs directory as described above.

## Information

_Author(s)_: François Serra, Davide Baù, David Castillo, Guillaume Filion, Marc A. Marti-Renom

_Date_: October 1st, 2016

_Last known good IMP version_: [![build info](https://salilab.org/imp/systems/?sysstat=12&branch=master)](http://salilab.org/imp/systems/) [![build info](https://salilab.org/imp/systems/?sysstat=12&branch=develop)](http://salilab.org/imp/systems/)

_Testable_: Yes.

_Parallelizeable_: Yes

_Citation_:
 - Serra, F., Baù, D., Filion, G., & Marti-Renom, M. A. (2016).
**Structural features of the fly chromatin colors revealed by automatic three-dimensional modeling.**
*bioRxiv*. `doi:10.1101/036764 <http://biorxiv.org/cgi/content/short/036764>`_
