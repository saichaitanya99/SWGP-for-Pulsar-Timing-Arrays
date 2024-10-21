# SWGP
Noise analysis code. This code has ability to sample and plot things as well depending on the arguments used. If you just want to plot, then you can also specify what all parameters to be sampled to create the pta object. If you give more than one pulsar, then the code will search for a common SW signal across all the pulsars by default. The hypermodel flag will compare the Nitu2024 and Susarla2024 models. More custom models will be added in the future. The number of hypermodel samples are fixed to 2e6 samples. The code will also save the reconstructed TOAs and DM data in the output directory.

The following are the current options: 

This document describes the command-line arguments available for the script. Each argument can be used to configure the script's behavior and specify input/output files. Please give proper paths for the parfile and timfile. Relative paths should be enough.

Arguments
-psrname (optional, str): Specify the name(s) of the pulsar(s). You can provide multiple names as needed.
-parfile (optional, str): Provide the path to the parfile(s) associated with the pulsar(s). Multiple parfiles can be specified.
-timfile (optional, str): Provide the path to the timfile(s) associated with the pulsar(s). You can specify multiple timfiles.
-out_dir (optional, str): Specify the path to the output directory. This argument is required if the intention is to perform noise analysis.
-nfit (optional, int, default: 0): Set the number of fits for libstempo. The default value is 0.
--just_plot (optional, bool): If this flag is set, the script will only plot the posteriors and the time domain reconstruction, skipping any fitting procedures.
-params_plot (optional, str): Provide the parameters you wish to plot. Multiple parameters can be specified. If this argument is omitted, all parameters will be plotted. The possible parameters include: efac, equad, red_noise, wn, dm_gp, gamma_sw, A_sw, and n_earth.
--plot_after_fitting (optional, bool): If set, the script will show how the residuals appear before the sampling begins. If phase wrapping is observed in the residuals, necessary actions may need to be taken.
--nofit (optional, bool): If this flag is set, the script will skip the fitting process for the provided parfile and timfile.
-chain_dir (optional, str): Specify the path to the chain directory if you only want to plot the posteriors and the time domain reconstruction.
-nsample (optional, int, default: 1e6): Set the number of samples for PTMCMC. The default value is 1 million samples. If you are using hypermodel, then the no of samples are defaulted to 2 million samples.
--dmgp (optional, bool): Use this flag to enable DMGP (Dispersion Measure Gaussian Process).
--dmgp_sample_nbins (optional, bool): Set this flag to sample the number of bins for DMGP. This option is only applicable when using DMGP.
-dmgp_nbins (optional, int, default: 50): Specify the number of bins for DMGP. The default value is 50.
--swsigma (optional, bool): Use this flag to enable SW Sigma, which utilizes the Nitu2024 model for solar wind.
--sw-sigma-max (optional, float, default: 10): Define the maximum solar wind sigma.
--sw-sigma-min (optional, float, default: 1e-3): Define the minimum solar wind sigma.
--swgp (optional, bool): Use this flag to enable SW GP (Solar Wind Gaussian Process), utilizing the Susarla2024 model for solar wind modeling.
-swgp_basis (optional, str, default: powerlaw): Specify the basis for SWGP. Options are powerlaw or spectrum, with powerlaw being the default.
--nocutoff (optional, bool): Use this flag if you do not wish to apply a cutoff for SWGP.
-cutoff (optional, float, default: 1.5): Specify the cutoff for SWGP. The default is 1.5 years, which means frequencies below 1/1.5 years will be used.
--swgp_sample_nbins (optional, bool): Set this flag to sample the number of bins for SWGP. This option is only applicable when using SWGP.
-swgp_nbins (optional, int, default: 30): Specify the number of bins for SWGP. The default value is 30.
--rn (optional, bool): Use this flag to enable red noise (RN).
-rn_nbins (optional, int, default: 30): Specify the number of bins for RN. The default value is 30.
--wn (optional, bool): Use this flag to enable white noise (WN).
--nesw (optional, bool): Use this flag to enable non-ecliptic solar wind (NE SW).
--no_plot (optional, bool): If set, the script will only sample the parameters without generating any plots.
--sims (optional, bool): Use this flag to enable simulations.
--no_sample (optional, bool): If this flag is set, the script will only display all the parameters and priors without performing any sampling.
--hypermodel (optional, bool): Use this flag for hypermodel comparisons. Currently, the script compares the Nitu2024 and Susarla2024 models. More custom models will be added in the future.