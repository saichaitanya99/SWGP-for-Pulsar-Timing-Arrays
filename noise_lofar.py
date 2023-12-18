import os
import subprocess
import numpy as np
import libstempo as LT
from enterprise.pulsar import Pulsar
from enterprise.signals import (gp_signals, 
                                white_signals, 
                                gp_priors, 
                                gp_bases, 
                                parameter, 
                                selections, 
                                signal_base)
from enterprise.signals import (deterministic_signals, gp_signals, parameter,
                                signal_base, utils)
from enterprise.signals.selections import Selection
from enterprise_extensions.chromatic.solar_wind import (solar_wind,createfourierdesignmatrix_solar_dm,solar_wind_block)
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import matplotlib.pyplot as plt
from enterprise_extensions.sampler import JumpProposal
import corner
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import libstempo.plot as LP
import scipy.linalg as sl

# Create the argument parser
parser = argparse.ArgumentParser(description='Noise analysis code. If you just want to plot, then you should also specify what all parameters to be sampled to create the pta object')

# Add the arguments
parser.add_argument('-psrname', required=True, type=str, help='Pulsar name')
parser.add_argument('-parfile', required=True, type=str, help='Parfile')
parser.add_argument('-timfile', required=True, type=str, help='timfile')
parser.add_argument('-out_dir', required=False, type=str, help='Provide the path to the output_dir. You need to provide one if the intention is noise analysis')
parser.add_argument('--just_plot', action='store_true', help='If you just want to plot the posteriors and the time domain reconstruction')
parser.add_argument('-chain_dir', required=False, help='Provide the path to chain directory to plot')
parser.add_argument('-nsample', required=False, type=int, default = 1e6, help='No of samples for PTMCMC. Default is 1 million samples')
parser.add_argument('--dmgp', action='store_true', help='Use this flag for dmgp')
parser.add_argument('--dmgp_nbins', action='store_true', help='Specify bins for DMGP')
parser.add_argument('--swgp', action='store_true', help='Use this flag for swgp')
parser.add_argument('--swgp_nbins', action='store_true', help='Specify bins for SWGP')
parser.add_argument('--rn', action='store_true', help='Use this flag for rn')
parser.add_argument('--rn_nbins', action='store_true', help='Specify bins for SWGP')
parser.add_argument('--wn', action='store_true', help='Use this flag for wn')
parser.add_argument('--nesw', action='store_true', help='Use this flag for nesw')
parser.add_argument('--sims', action='store_true', help='Use this flag for simulations')


# Parse the arguments
args = parser.parse_args()


if args.just_plot:
    if args.chain_dir==False:
        print("You have to provide the chain directory to plot!")
        exit(0)

def solar_wind_b(n_earth=None, ACE_prior=False, include_swgp=True,
                     swgp_prior=None, swgp_basis=None, Tspan=None):
    """
    Returns Solar Wind DM noise model. Best model from Hazboun, et al (in prep)
        Contains a single mean electron density with an auxiliary perturbation
        modeled using a gaussian process. The GP has common prior parameters
        between all pulsars, but the realizations are different for all pulsars.
        Solar Wind DM noise modeled as a power-law with 30 sampling frequencies

    :param n_earth:
        Solar electron density at 1 AU.
    :param ACE_prior:
        Whether to use the ACE SWEPAM data as an astrophysical prior.
    :param swgp_prior:
        Prior function for solar wind Gaussian process. Default is a power law.
    :param swgp_basis:
        Basis to be used for solar wind Gaussian process.
        Options includes ['powerlaw'.'periodic','sq_exp']
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar. Default is to use 15
        frequencies (1/Tspan,15/Tspan).

    """

    if n_earth is None and not ACE_prior:
        n_earth = parameter.Uniform(0, 25)('n_earth')
    elif n_earth is None and ACE_prior:
        n_earth = ACE_SWEPAM_Parameter()('n_earth')
    else:
        pass

    deter_sw = solar_wind(n_earth=n_earth)
    mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
    sw_model = mean_sw

    if include_swgp:
        if swgp_basis == 'powerlaw':
            # dm noise parameters that are common
            log10_A_sw = parameter.Uniform(-12, 1)
            gamma_sw = parameter.Uniform(-3, 9)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)

            if Tspan is not None:
                freqs = np.linspace(1/Tspan, 30/Tspan, 30)
                freqs = freqs[1/freqs > 1.5*yr_in_sec]
                sw_basis = createfourierdesignmatrix_solar_dm(modes=freqs)
            else:
                print("100 bins for SWGP are being used")
                sw_basis = createfourierdesignmatrix_solar_dm(nmodes=100,
                                                              Tspan=Tspan)
        elif swgp_basis == 'periodic':
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            sw_basis = gpk.linear_interp_basis_dm(dt=6*86400)
            sw_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                                           log10_ell=log10_ell,
                                           log10_gam_p=log10_gam_p,
                                           log10_p=log10_p)
        elif swgp_basis == 'sq_exp':
            # squared-exponential GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)

            sw_basis = gpk.linear_interp_basis_dm(dt=6*86400)
            sw_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma,
                                        log10_ell=log10_ell)

        gp_sw = gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')
        sw_model += gp_sw

    return sw_model






def get_b(d, TNT, phiinv):
	# Taken from la_forge: https://github.com/nanograv/la_forge/tree/main/la_forge
	
	Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
	try:
		u, s, _ = sl.svd(Sigma)
		mn = np.dot(u, np.dot(u.T, d)/s)
		Li = u * np.sqrt(1/s)
	except np.linalg.LinAlgError:
		Q, R = sl.qr(Sigma)
		Sigi = sl.solve(R, Q.T)
		mn = np.dot(Sigi, d)
		u, s, _ = sl.svd(Sigi)
		Li = u * np.sqrt(1/s)

	return mn + np.dot(Li, np.random.randn(Li.shape[0]))

def get_tdelay_from_chains(pta, psr, ch, pars, 
						   ipsr=0, 
						   nsamples=100, ch_idxs=None, 
						   signames=['all'], 
						   separe_signals=True,
						   plttypes=['ptmcmc'],
						   verbose=True):	
	if ch_idxs is None:
		ch_idxs = [np.random.choice(len(ch)) for iii in range(nsamples)]
	
	if list(pars) != pta.param_names:
		print("WARNING: parameter names don't correspond with the ones in the PTA object created from strmodel.")
		print("\npars:")
		print("\n".join(pars))
		print("\nPTA param names:")
		print("\n".join(pta.param_names))
		
	sig_idxs = {}
	for s, idx in pta._signalcollections[ipsr]._idx.items():
		sig_idxs.update({s.signal_id:idx})
	
	if separe_signals:
		delays = np.zeros((len(signames), len(ch_idxs), len(psr.toas)))
	else:
		delays = np.zeros((len(ch_idxs), len(psr.toas)))

	pta_signals = pta._signalcollections[ipsr]._signals

	for i, ch_idx in enumerate(ch_idxs):
		if verbose:
			print("%s on %s"%(i+1, len(ch_idxs)), end="\r")
			
		#post_sample = {parn:val for parn, val in zip(pta.param_names, ch[ch_idx,:])}
		post_sample = pta.map_params(ch[ch_idx,:])

		# Get matrices, basis functions and indexes for the GP signals
		TNrs = pta.get_TNr(post_sample) # [len(basis functions)] * len(psrs)
		TNTs = pta.get_TNT(post_sample) # [len(basis function) x len(basis function)] * len(psrs)
		#print(post_sample)
		phiinvs = pta.get_phiinv(post_sample, logdet=False)  # [len(basis functions)] * len(psrs)
		Ts = pta.get_basis(post_sample) # [len(ToAs) x len(basis function)] * len(psrs)
		w = get_b(TNrs[ipsr], TNTs[ipsr], phiinvs[ipsr]) # basis functions

		for j, signame in enumerate(signames):
			delay = 0

			if signame=='all':
				# Find all signals in pta object
				pta_sig = [s for s in pta_signals]
			else:
				# Find related signal in pta object
				#pta_sig = [s for s in pta_signals if signame in s.signal_id]

				# Strict choice
				pta_sig = [s for s in pta_signals if signame == s.signal_id]
			
			for s in pta_sig:
				# Add deterministic signals
				if s.signal_type == 'deterministic':
					delay += s.get_delay(post_sample)
					
				# Add GP signals
				if s.signal_type == 'basis':
					idx = sig_idxs[s.signal_id]
					delay += np.dot(Ts[ipsr][:, idx], w[idx])

			if plttypes[j]=="dm":
				delay *= psr.freqs**2*2.41e-4
			elif plttypes[j]=="sv":
				delay *= psr.freqs**4
			
			if separe_signals:
				delays[j, i, :] = delay
			else:
				delays[i, :] += delay
	
	return delays




# Set data

# Create Libstempo pulsar object
psrname = args.psrname 


outdir = args.out_dir
if args.just_plot and args.out_dir==None:
    outdir= "chain_%s"%psrname
else:
    outdir+= "chain_%s"%psrname
plotname = "corner_%s"%psrname
recons_plot = "recons_%s"%psrname
parfile=args.parfile  #parfile = "%s.par"%psrname
timfile=args.timfile   #timfile = "J0034-0534_ToAs.tim"#"%s_lofarpglow_filtered.tim"%psrname
ltpsr = LT.tempopulsar(parfile, timfile,maxobs=45000)




### Fit for the Timing Model
Nfit = 10
for i in range(Nfit):
    ltpsr.fit()


    
    
### Create enterprise.Pulsar object from tempopulsar object
psr = Pulsar(ltpsr)

print("libstempo fitting done")



# Set noise model

### Create enterprise signal object
##############################
## Timing model marginalization
s = gp_signals.TimingModel()

##############################
## White noise

backend = selections.Selection(selections.by_backend)
efac = parameter.Uniform(0.1, 5.0)
equad = parameter.Uniform(-8, -2) #changed
#selection = selections.Selection(selections.no_selection)
#efac = parameter.Constant()
#wn = white_signals.MeasurementNoise(efac=efac, selection=selection)
efeq = white_signals.MeasurementNoise(efac=efac, selection=backend, name=None) # EFAC
efeq += white_signals.TNEquadNoise(log10_tnequad=equad, selection=backend, name=None) # EQUAD
if args.wn:
    s += efeq
    outdir += "_wn"
    plotname += "_wn"
    recons_plot += "_wn"

##############################

## Achromatic Red noise
Nbins = 30

# Simple powerlaw PSD

log10_A = parameter.Uniform(-20, -8)
gamma = parameter.Uniform(0, 7)
pl = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)

rn = gp_signals.FourierBasisGP(pl, components=Nbins, name='red_noise')
if args.rn:
    s += rn
    outdir += "_rn"
    plotname += "_rn"
    recons_plot += "_rn"

##########################


if args.swgp and args.nesw:
    s += solar_wind_b(ACE_prior=False, include_swgp=True,swgp_prior='powerlaw', swgp_basis='powerlaw',Tspan=None)
    outdir += "_nesw_swgp"
    plotname += "_nesw_swgp"
    recons_plot += "_nesw_swgp"
elif args.nesw:
    s += solar_wind_b(ACE_prior=False, include_swgp=False,swgp_prior='powerlaw', swgp_basis='powerlaw',Tspan=None)
    outdir += "_nesw"
    plotname += "_nesw"
    recons_plot += "_nesw"



##############################

## DM variations
Nbins = 50

# Simple powerlaw PSD
log10_A = parameter.Uniform(-18, -4)
gamma = parameter.Uniform(0, 7)
pl = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)

dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=Nbins)
dmgp = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')
if args.dmgp:
    s += dmgp
    outdir += "_dmgp"
    plotname += "_dmgp"
    recons_plot += "_dmgp"


# Set PTA object (used to compute Likelihood and prior)

### Create PTA object, used to compute likelihood and prior values
model = [s(psr)]
pta = signal_base.PTA(model)
params = {"%s_efac"%psrname: 1.}
pta.set_default_params(params)

# Check LH and prior vals

# Print parameters and priors
for i in pta.params:
    print(i)

# Get Likelihood and prior value from a vector of param values randomly sampled
x0 = np.hstack([p.sample() for p in pta.params])
print("x0:\n")
for i,p in enumerate(pta.param_names):
    print("{0:<37s} : {1:6.3f}".format(p,x0[i]))

print("\nLH(x0) = %.3f"%(pta.get_lnlikelihood(x0)))
print("Prior(x0) = %.3f"%(pta.get_lnprior(x0)))


if args.just_plot==False:
    # Set sampler

    ### Sample parameters, using PTMCMCSampler
    resume=False # Use it to resume existing chain


    #outdir = "./chain_J0034-0534_finaldatarun_allnoise"
    if not os.path.isdir(outdir):
        subprocess.check_output('mkdir -p '+outdir, shell=True)
    Nsamples = int(args.nsample)
    np.savetxt(outdir+'/pars.txt', pta.param_names, fmt='%s')

    ###### set-up the sampler
    x0 = np.hstack([p.sample() for p in pta.params])
    ndim = len(x0)

    ###### initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution

    ## Create sampler object
    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=outdir, resume=resume)

    # Set Jump Proposals
    jp = JumpProposal(pta, snames=None)

    sampler.addProposalToCycle(jp.draw_from_prior, 50)

    sel_sig = {"timing_model":20,
                   "rn_":10, 
                   "red_noise_":10, 
                   "dm_":10, 
                   "fcn_":10, 
                   "cn_":10, 
                   "expd":10, 
                   "yearly":10, 
                   "crs":30,
                   "mass":10,
                   "orb_elements":30,
                   "frame_drift_rate":30,
                   "egp":30,"gp_sw":10,"n_earth":10}

    for s,n in sel_sig.items():
        if any([s in p for p in pta.param_names]):
            sampler.addProposalToCycle(jp.draw_from_par_prior(s), n)

    #sampler.addProposalToCycle(pta.draw_from_signal_prior, 50)

    # Launch sampling !

    sampler.sample(x0, Nsamples, SCAMweight=40, AMweight=15, DEweight=60)


    xs = {par.name: par.sample() for par in pta.params}
    print(xs)






##PLotting script here####

if args.just_plot:
    outdir=args.chain_dir
    
ch = np.loadtxt(f'{outdir}/chain_1.txt')

with open(f'{outdir}/pars.txt') as f1:
    xs1 = f1.readlines()


xs11 = [i.split('\n',1) for i in xs1]
pars = ([xs11[i][0] for i in range(len(xs11))])
idxs = np.arange(len(pars))


burn = int(0.25 * ch.shape[0])

fig = corner.corner(ch[burn:, idxs],bins =30, labels=[i.replace(psrname+"_", "") for i in pars],hist_kwargs={'density':True}, plot_datapoints=True, color='r',show_titles=True)

plt.show()
plt.savefig(f'{outdir}/{plotname}')


####Recons script#####


ch_idxs = 100*[np.argmax(ch[:,-3])] # Take 500 



t = get_tdelay_from_chains(pta, psr, ch, pars, ch_idxs=ch_idxs)


mask = np.argsort(ltpsr.toas())
fig = plt.figure(figsize=(8,6))
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4,1], hspace=0.01)
ax0 = fig.add_subplot(spec[0])
ax0.grid()
ax0.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax0.set_ylabel("Residuals", fontsize=11, fontweight='bold')
ax0.get_yaxis().get_major_formatter().set_scientific(False)
ax0.yaxis.set_label_position("left")

for i in range(len(t[0])):
    ax0.plot(ltpsr.toas()[mask], t[0,i,:], color='coral', alpha=.3)
means = np.mean(t[0,:,:], axis=0)

ax0.errorbar(ltpsr.toas(),ltpsr.residuals(),yerr=1e-6*ltpsr.toaerrs,fmt='.',alpha=0.2,label="Original TOAs")
ax0.plot(ltpsr.toas()[mask], means, color='r', lw=2, alpha=.3, label="Recovered signal")
ax0.axhline(0., ls=':', c='k', lw=3)
ax0.legend()


ax1 = fig.add_subplot(spec[1])
ax1.grid()
ax1.set_ylabel("Recovered-original", fontsize=11, fontweight='bold')
ax1.set_xlabel('MJD', fontsize=11, fontweight='bold')
ax1.plot(ltpsr.toas()[mask], means-ltpsr.residuals(), color='r', lw=2, alpha=.3, label="Difference")
plt.show()
plt.savefig(f'{outdir}/{recons_plot}')
















