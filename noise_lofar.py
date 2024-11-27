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
from enterprise_extensions.chromatic.solar_wind import (solar_wind,createfourierdesignmatrix_solar_dm,ACE_SWEPAM_Parameter)
from enterprise_extensions import dropout as drop
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import model_utils 
import matplotlib.pyplot as plt
from enterprise_extensions.sampler import JumpProposal
import corner
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import warnings
import libstempo.plot as LP
import scipy.linalg as sl
import ast
from functools import partial
import scipy.optimize
from astropy.coordinates import Angle
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates import get_sun
from astropy.time import Time
from enterprise_extensions.blocks import white_noise_block, red_noise_block, dm_noise_block, chromatic_noise_block
from enterprise_extensions import hypermodel


warnings.filterwarnings("ignore")



# Create the argument parser
parser = argparse.ArgumentParser(description='Noise analysis code.' 
                                 ' This code has ability to sample and plot things as well depending on the arguments used. '
                                 'If you just want to plot, then you can also specify what all parameters to be sampled to create the pta object.'
                                 'If you give more than one pulsar, then the code will search for a common SW signal across all the pulsars by default.'
                                 ' The hypermodel flag will compare the Nitu2024 and Susarla2024 models. More custom models will be added in the future. The number of hypermodel samples are fixed to 2e6 samples. '
                                 'The code will also save the reconstructed TOAs and DM data in the output directory. ')

# Add the arguments
parser.add_argument('-psrname', required=False, type=str, nargs='+', help='Pulsar name(s)')
parser.add_argument('-parfile', required=False, type=str, nargs='+', help='Parfile(s)')
parser.add_argument('-timfile', required=False, type=str, nargs='+', help='timfile(s)')
parser.add_argument('-out_dir', required=False, type=str, help='Provide the path to the output_dir. You need to provide one if the intention is noise analysis')
parser.add_argument('-nfit', required=False, type=int, default=0, help='No of fits for libstempo. Default is 0.')
parser.add_argument('--just_plot', action='store_true', help='If you just want to plot the posteriors and the time domain reconstruction')
parser.add_argument('-params_plot', required=False, type=str, nargs='+', help='Provide the parameters to be plotted. You can provide multiple parameters to be plotted. If you do not provide this argument, then all the parameters will be plotted. The possible parameters are efac, equad, red_noise, wn, dm_gp, gp_sw, n_earth.')
parser.add_argument('--plot_after_fitting', action='store_true', help='If you want to see how the residuals look before the start of sampling. If you see phase wrapping in the residuals, please take necessary action.')
parser.add_argument('--nofit', action='store_true', help='If you do not want to fit the parfile and timfile')
parser.add_argument('-chain_dir', required=False, help='Provide the path to the chain directory if you just want to plot the posteriors and the time domain reconstruction')
parser.add_argument('-nsample', required=False, type=int, default = 1e6, help='No of samples for PTMCMC. Default is 1 million samples. If you are using hypermodel, then the no of samples are defaulted to 2 million samples.')
parser.add_argument('--dmgp', action='store_true', help='Use this flag for dmgp')
parser.add_argument('--dmgp_sample_nbins', action='store_true', help='Use this flag for dmgp_sample_nbins. This will also sample the number of bins for dmgp. You can use this flag only if you are using dmgp')
parser.add_argument('-dmgp_nbins', required=False, default=50, type=int, help='Specify bins for DMGP, Default is 50')
parser.add_argument('--swsigma', '--solar-wind',action='store_true', help='Use this flag for swsigma. This will use the Nitu2024 model for solar wind')
parser.add_argument('--sw-sigma-max', type=float, default=10, help='Max solar wind sigma')
parser.add_argument('--sw-sigma-min', type=float, default=1e-3, help='Min solar wind sigma')
parser.add_argument('--swgp', action='store_true', help='Use this flag for swgp. This will use Susarla2024 model for modeling solar wind')
parser.add_argument('-swgp_basis', required=False, default='powerlaw', type=str, help='Specify basis for SWGP, powerlaw or spectrum, Default is powerlaw')
parser.add_argument('--nocutoff', action='store_true', help='Use this flag if you do not want to use cutoff for SWGP')
parser.add_argument('-cutoff', required=False, default=1.5, type=float, help='Specify cutoff for SWGP, Default is 1.5 yr, so the frequencies below 1/1.5yr will be used')
parser.add_argument('--swgp_sample_nbins', action='store_true', help='Use this flag for swgp_sample_nbins. This will also sample the number of bins for swgp. You can use this flag only if you are using swgp')
parser.add_argument('-swgp_nbins', required=False, default=30, type=int, help='Specify bins for SWGP, Default is 30')
parser.add_argument('--rn', action='store_true', help='Use this flag for rn')
parser.add_argument('-rn_nbins', required=False, default=30, type=int, help='Specify bins for SWGP, Default is 30')
parser.add_argument('--wn', action='store_true', help='Use this flag for wn')
parser.add_argument('--nesw', action='store_true', help='Use this flag for nesw')
parser.add_argument('--no_plot', action='store_true', help='Only sample the parameters and do not plot')
parser.add_argument('--sims', action='store_true', help='Use this flag for simulations')
parser.add_argument('--no_sample', action='store_true', help='Only show all the parameters and priors')
parser.add_argument('--hypermodel', action='store_true', help='Use this flag for hypermodel. Currently this code only compares Nitu2024 and Susarla2024 models. More custom models will be added in the future')



# Parse the arguments
args = parser.parse_args()

yr_in_sec = 365.25*24*3600


if args.swgp and args.swsigma:
    print("You have to choose either swgp or swsigma, not both!")
    exit(0)




def load_chain_and_params(outdir):
    """Loads MCMC chain and parameter names from output directory."""
    ch = np.loadtxt(f'{outdir}/chain_1.txt')
    with open(f'{outdir}/pars.txt') as f:
        params = [line.strip() for line in f.readlines()]
    return ch, params


def read_psrname_from_args(outdir):
    """Reads pulsar name from command line arguments."""
    with open(f'{outdir}/command_line_args.txt') as f:
        for line in f:
            if line.startswith('psrname'):
                return ast.literal_eval(line.split(":")[1].strip())
    return None


def get_plot_labels(params, psrname, args):
    """Generates plot labels based on selected parameters."""
    if args.params_plot:
        matching_params = [p for p in params if any(param in p for param in args.params_plot)]
        matching_indices = [params.index(p) for p in matching_params]
        return matching_params,matching_indices
    else:
        if len(psrname) > 1:
            return params, np.arange(len(params))
        else:
            return [p.replace(psrname[0] + "_", "") for p in params], np.arange(len(params))


def plot_corner(ch, burn, idxs, labels, outdir, plotname):
    """Plots corner plot using MCMC chain data."""
    print(labels)
    print("Plotting corner plot")
    fig = corner.corner(ch[burn:, idxs], bins=30, labels=labels, hist_kwargs={'density': True}, plot_datapoints=True, color='r', show_titles=True)
    plt.savefig(f'{outdir}/{plotname}.png')


def plot_residuals(ltpsr, t, outdir, recons_plot):
    """Plots TOA residuals and time-domain reconstructions."""
    mask = np.argsort(ltpsr.toas)
    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.01)
    
    ax0 = fig.add_subplot(spec[0])
    ax0.grid()
    ax0.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax0.set_ylabel("Residuals", fontsize=11, fontweight='bold')

    for i in range(len(t[0])):
        ax0.plot(ltpsr.toas[mask], t[0, i, :], color='coral', alpha=0.3)
    
    means = np.mean(t[0, :, :], axis=0)
    ax0.errorbar(ltpsr.toas, ltpsr.residuals, yerr=1e-6 * ltpsr.toaerrs, fmt='.', alpha=0.2, label="Original TOAs")
    ax0.plot(ltpsr.toas[mask], means, color='r', lw=2, alpha=0.3, label="Recovered signal")
    ax0.axhline(0., ls=':', c='k', lw=3)
    ax0.legend()

    ax1 = fig.add_subplot(spec[1])
    ax1.grid()
    ax1.set_ylabel("Recovered-original", fontsize=11, fontweight='bold')
    ax1.set_xlabel('MJD', fontsize=11, fontweight='bold')
    ax1.plot(ltpsr.toas[mask], means - ltpsr.residuals, color='r', lw=2, alpha=0.3, label="Difference")
    
    plt.savefig(f'{outdir}/{recons_plot}.png')


def save_reconstructed_toas(outdir, psrname, ltpsr, means, toa_rec_err, DM_SW, DM_COMB):
    """Saves reconstructed TOAs and DM data."""
    toa_rec = np.column_stack((ltpsr.toas(), means, toa_rec_err))
    np.savetxt(f'{outdir}/{psrname[0]}_TOA_rec.txt', toa_rec, delimiter=' ')

    dm_rec_sw =np.column_stack((ltpsr.toas(), np.mean(DM_SW[:,:], axis=0),np.std(DM_SW[:,:], axis=0)))
    dm_rec_comb = np.column_stack((ltpsr.toas(), np.mean(DM_COMB[:,:], axis=0),np.std(DM_COMB[:,:], axis=0)))
    np.savetxt(f'{outdir}/{psrname[0]}_DM_SW.txt', dm_rec_sw, delimiter=' ')
    np.savetxt(f'{outdir}/{psrname[0]}_DM_COMB.txt', dm_rec_comb, delimiter=' ')


def plotting(outdir=None, pta=None, psr=None, plotname=None, recons_plot=None):
    """Main plotting function that handles MCMC results and TOA reconstruction."""
    
    # Read pulsar name and determine plot names
    psrname = read_psrname_from_args(outdir)
    if len(psrname) > 1:
        plotname = "corner_common_sw"
        recons_plot = "recons_common_sw"
    else:
        plotname = f"corner_{psrname[0]}"
        recons_plot = f"recons_{psrname[0]}"

    # Load chain and parameters
    ch, params = load_chain_and_params(outdir)
    burn = int(0.25 * ch.shape[0])

    # Get plot labels
    labels,idxs = get_plot_labels(params, psrname, args)

    # Plot corner plot
    plot_corner(ch, burn, idxs=idxs, labels=labels, outdir=outdir, plotname=plotname)

    # Time-domain reconstruction (handling multiple scenarios)
    ch_idxs = 100 * [np.argmax(ch[:, -3])]
    t = get_tdelay_from_chains(pta, psr, ch, params, ch_idxs=ch_idxs)

    # Reconstruct DM (handling solar wind and noise models)
    if args.swgp and args.nesw:
        DM_sw = get_tdelay_from_chains(pta, psr, ch, params, signames=['gp_sw', 'n_earth'], plttypes=['dm', 'dm'], ch_idxs=ch_idxs,separe_signals=False)
        DM_comb = get_tdelay_from_chains(pta, psr, ch, params, signames=['gp_sw', 'n_earth', 'dm_gp', 'linear_timing_model'], plttypes=['dm', 'dm', 'dm', 'dm'], ch_idxs=ch_idxs, separe_signals=False)

    elif args.swsigma and args.nesw:
        DM_sw = get_tdelay_from_chains(pta, psr, ch, params, signames=['SW_sigma', 'n_earth'], plttypes=['dm', 'dm'], ch_idxs=ch_idxs, separe_signals=False)
        DM_comb = get_tdelay_from_chains(pta, psr, ch, params, signames=['SW_sigma', 'n_earth', 'dm_gp', 'linear_timing_model'], plttypes=['dm', 'dm', 'dm', 'dm'], ch_idxs=ch_idxs, separe_signals=False)

    # Plot residuals and reconstruction
    plot_residuals(psr, t, outdir, recons_plot)

    # Save TOAs and DM data
    save_reconstructed_toas(outdir, psrname, ltpsr, np.mean(t[0, :, :], axis=0), np.std(t[0, :, :], axis=0), DM_sw, DM_comb)



def common_solar_wind(n_earth=None, ACE_prior=False, include_swgp=True,
                     swgp_prior=None, swgp_basis=None, Tspan=None,include_n_earth=True):
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
    sw_model = None
    if include_n_earth:
        if n_earth is None and not ACE_prior:
            n_earth = parameter.Uniform(0, 25)('n_earth')
            #n_earth = parameter.Uniform(0,20,size=n_earth_bins)('n_earth')
        elif n_earth is None and ACE_prior:
            n_earth = ACE_SWEPAM_Parameter()('n_earth')
        else:
            pass

        deter_sw = solar_wind(n_earth=n_earth)#, n_earth_bins=n_earth_bins)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
        sw_model = mean_sw

    cutoff = args.cutoff
    if include_swgp:

        if swgp_basis == 'spectrum':
            # dm noise parameters that are common
            if Tspan is not None:
                freqs = np.linspace(1/Tspan, args.swgp_nbins/Tspan, args.swgp_nbins)
                freqs = freqs[1/freqs > cutoff*yr_in_sec]
                log10_rho = parameter.Uniform(-4,9, size=len(freqs))
                print(f"{len(freqs)} bins for SWGP are being used")
                sw_basis = createfourierdesignmatrix_solar_dm(modes=freqs)
            else:
                log10_rho = parameter.Uniform(-10, 1, size=15)
                sw_basis = createfourierdesignmatrix_solar_dm(nmodes=15,
                                                              Tspan=Tspan)
                
            sw_prior = gp_priors.free_spectrum(log10_rho=log10_rho)


        if swgp_basis == 'powerlaw':
            if len(psrname)>1:
                log10_A_sw = parameter.Uniform(-12, 1)('log10_A_sw')
                gamma_sw = parameter.Uniform(-6, 5)('gamma_sw')
            else:
                log10_A_sw = parameter.Uniform(-12, 1)
                gamma_sw = parameter.Uniform(-6, 5)
            

            if Tspan is not None:
                freqs = np.linspace(1/Tspan, args.swgp_nbins/Tspan, args.swgp_nbins)
                if args.nocutoff==False:
                    freqs = freqs[1/freqs > cutoff*yr_in_sec]
                print(f"{len(freqs)} bins for SWGP are being used")
                sw_basis = createfourierdesignmatrix_solar_dm(modes=freqs)
            else:
                print(f"{args.swgp_nbins} bins for SWGP are being used")
                sw_basis = createfourierdesignmatrix_solar_dm(nmodes=args.swgp_nbins)

            if args.swgp_sample_nbins:
                k_dropbin = parameter.Uniform(2, len(freqs)+1)
                sw_prior = drop.dropout_powerlaw(log10_A=log10_A, gamma=gamma, k_drop=None,k_dropbin=k_dropbin)
            else:
                sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
        
            
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


def fcos_full_nitu(x, phi, eclat):
    # function to fit ctheta
    return np.cos(eclat) * np.cos(2*np.pi*x/365.25 - phi)


def solar_angle(coords_psr, mjd):

    #RA Dec of the pulsar in format e.g.: 10h22m57.9992s +10d01m52.78s)                                                                                                                                            

    t = Time(mjd, format='mjd')
    sunpos = get_sun(t)
    sep = sunpos.separation(coords_psr)

    return sep

def ifunc_nitu(xi, i, x):
    y = np.zeros_like(xi)
    y[i] = 1
    return np.interp(x.astype(np.float64), xi.astype(np.float64), y.astype(np.float64))



def setup_psr_nitu(psr,forb=False):

    # I *think* the psr.earth_ssb[:,3:5] etc are velocity components and I only need [:,:3]
    t2psr = psr.t2pulsar


    rsa = -t2psr.sun_ssb[:, :3] + t2psr.earth_ssb[:, :3] + t2psr.observatory_earth[:, :3]

    # r = np.sqrt(rsa * rsa)
    r = np.empty(rsa.shape[0])
    for j in range(rsa.shape[0]):
        r[j] = np.sqrt(np.dot(rsa[j], rsa[j]))

    pos = t2psr.psrPos  # this is probably corrected for velocity already since it only has size 3 (not 6)

    # ctheta = (pos * rsa) / r
    ctheta = np.empty(rsa.shape[0])
    for j in range(rsa.shape[0]):
        ctheta[j] = np.dot(pos[j], rsa[j]) / r[j]

    ''' From dm_delays.C:

    psr[p].obsn[i].freqSSB = freqf; /* Record observing frequency in barycentric frame */ (in Hz)

    '''

    freqf = t2psr.ssbfreqs()  # observing freq in barycentre frame, in Hz

    ''' From tempo2.h:

    #define AU_DIST     1.49598e11           /*!< 1 AU in m  
    #define DM_CONST    2.41e-4
    #define DM_CONST_SI 7.436e6              /*!< Dispersion constant in SI units            */
    #define SPEED_LIGHT          299792458.0 /*!< Speed of light (m/s)                       */

    '''
    AU_DIST = 1.49598e11
    DM_CONST_SI = 7.436e6
    SPEED_LIGHT = 299792458.0

    # The symmetrical, spherical solar wind, depending on observing frequency.

    spherical_solar_wind = 1.0e6 * AU_DIST * AU_DIST / SPEED_LIGHT / DM_CONST_SI * \
                           np.arccos(ctheta) / r / np.sqrt(1.0 - ctheta * ctheta) / freqf / freqf

    psr.spherical_solar_wind = spherical_solar_wind[psr.isort]

    psr.ctheta = ctheta[psr.isort]

    day_per_year = 365.25

    toas_mjds = t2psr.stoas
    
    # Fit a cos(2pi *t_mjd/day_per_year + phase) to ctheta, 
    # obtain phase and find the actual minimum of ctheta, i.e. the solar conjuction

    mjds = toas_mjds.astype(np.float64)


    # get ELAT, i.e. the max approach angle to the sun


    if 'RAJ' in t2psr.pars():
        
        coord_long = Angle(t2psr.vals()[t2psr.pars().index('RAJ')], unit=units.rad)
        coord_lat = Angle(t2psr.vals()[t2psr.pars().index('DECJ')], unit=units.rad)
        frame_psr = 'fk5'

    else:
        
        coord_long = Angle(t2psr.vals()[t2psr.pars().index('ELONG')], unit=units.rad)
        coord_lat = Angle(t2psr.vals()[t2psr.pars().index('ELAT')], unit=units.rad)
        frame_psr = 'barycentrictrueecliptic'

    coords_psr = SkyCoord(coord_long, coord_lat, frame=frame_psr)
    elat = coords_psr.barycentrictrueecliptic.lat



    fcos = partial(fcos_full_nitu, eclat=elat)
    popt, _ = scipy.optimize.curve_fit(fcos, mjds, ctheta, bounds=([0], [2*np.pi]))
    # phase = popt[0]

    # find the first APPROXIMATE conjuction in the data at theta = pi-ELAT
    Nfirstconj = int(mjds.min()/365.25 - 0.5 - (popt[0])/2/np.pi) + 1
    toa_solconj = 365.25 * (Nfirstconj + 0.5 + (popt[0])/2/np.pi)


    # get all the conjuctions before and after this 'minimum' one in the range of the data
    nlower = (toa_solconj - mjds.min()) / day_per_year
    nhigher = (mjds.max() - toa_solconj) / day_per_year

    conjunction_toas_mjds = np.concatenate((toa_solconj - np.arange(1, int(nlower) + 1) * day_per_year, [toa_solconj], toa_solconj + np.arange(1, int(nhigher) + 1) * day_per_year ))
    conjunction_toas_mjds.sort()
    nconj = len(conjunction_toas_mjds)

    # Now zoom in around these approximate conjunctions to find the conjunction MJD to 0.01 days:

    real_conjunction_toas_mjds = np.zeros_like(conjunction_toas_mjds)

    for i, c in enumerate(conjunction_toas_mjds):

        close_t = np.linspace(c-10, c+10, num=2000)
        close_ctheta = (-1) * np.cos(solar_angle(coords_psr, close_t))
        real_conjunction_toas_mjds[i] = close_t[np.argmin(close_ctheta)] 


    psr.sphconj_mjd = real_conjunction_toas_mjds.astype(np.float64)

    if forb:
        return psr.spherical_solar_wind, psr.sphconj_mjd



@signal_base.function
def solar_wind_basis_nitu(toas):

    psr = Pulsar(ltpsr,drop_t2pulsar=False)
    spherical_solar_wind, sphconj_mjd = setup_psr_nitu(psr,forb=True)

    nconj = len(sphconj_mjd)

    sec_per_day = 24 * 3600
    ifunc_comp = np.empty((nconj, len(toas)))

    for i in range(nconj):

        ifunc_comp[i] = ifunc_nitu(sphconj_mjd * sec_per_day, i, toas) * spherical_solar_wind

    solar_wind_basis = ifunc_comp

    solar_wind_basis = solar_wind_basis.T

    return solar_wind_basis, sphconj_mjd * sec_per_day



def solar_wind_basis_GP_nitu(priors, selection=Selection(selections.no_selection), name=''):

    basis = solar_wind_basis_nitu()
    BaseClass = gp_signals.BasisGP(priors, basis, selection=selection, name=name)

    class solar_wind_basis_class(BaseClass):
        signal_type = 'basis'
        signal_name = 'solar wind'
        signal_id = 'solar_wind_' + name if name else 'solar_wind'

    return solar_wind_basis_class



@signal_base.function
def sw_priors_nitu(t, sw_sigma=1):
    """
    This is the priors on our solar wind basis function

    Probably this is constant? Or maybe there is an option for a solar cycle one?
    would be in np.ones_like(t)
    :return:
    """

    return np.ones_like(t) * sw_sigma**2


def setup_model_nitu(psr, par):
    sw_model_nitu=None

    setup_psr_nitu(psr)

    sw_sigma = parameter.Uniform(args.sw_sigma_min, args.sw_sigma_max)("SW_sigma")
    
    if not args.hypermodel:
        with open(par) as f:
            parfile = f.readlines()
        # Add the SW_sigma parameter to the par file
        parfile.append("NE_SW_IFUNC 0 1\n")

        for conj_toa in psr.sphconj_mjd:
            parfile.append("_NE_SW {:.2f}\n".format(conj_toa))



    sw_model_nitu = solar_wind_basis_GP_nitu(priors=sw_priors_nitu(sw_sigma=sw_sigma))


    return sw_model_nitu









def save_args_to_file(args, filename='command_line_args.txt'):
    """Saves command-line arguments to a file."""
    args_dict = vars(args)
    with open(filename, 'w') as file:
        for arg, value in args_dict.items():
            file.write(f"{arg}: {value}\n")


def initialize_pulsars(parfile, timfile, psrname):
    """Creates libstempo pulsar objects."""
    if len(psrname) > 1:
        return [LT.tempopulsar(parfile[i], timfile[i], maxobs=100000) for i in range(len(psrname))]
    else:
        return LT.tempopulsar(parfile[0], timfile[0], maxobs=100000)


def fit_timing_model(ltpsr, psrname, Nfit, plot_after_fitting):
    """Fits timing model and optionally plots residuals."""
    print(f"Fitting for the timing model {Nfit} times")

    if len(psrname) > 1:
        for psr in ltpsr:
            for _ in range(Nfit):
                psr.fit()
    else:
        for i in range(Nfit):
            print(f"Fitting iteration {i}")
            ltpsr.fit()

    if plot_after_fitting:
        print("Plotting residuals after fitting.")
        if len(psrname) > 1:
            for psr in ltpsr:
                LP.plotres(psr)
                plt.show()
        else:
            LP.plotres(ltpsr)
            plt.show()

    print("libstempo fitting done")


def create_enterprise_pulsars(ltpsr, psrname):
    """Creates enterprise.Pulsar objects."""
    if len(psrname) > 1:
        return [Pulsar(psr) for psr in ltpsr]
    else:
        return Pulsar(ltpsr)


def setup_outdir(psrname, outdir, just_plot=False):
    """Sets up the output directory and plot/reconstruction filenames."""
    if just_plot:
        if outdir is None:
            outdir = f"chain_{psrname}"
        plotname, recons_plot = f"corner_{psrname}", f"recons_{psrname}"
    else:
        if len(psrname) > 1:
            outdir += "/chain_common_sw"
            plotname, recons_plot = "corner_common_sw", "recons_common_sw"
        else:
            outdir += f"/chain_{psrname[0]}"
            plotname, recons_plot = f"corner_{psrname[0]}", f"recons_{psrname[0]}"
    return outdir, plotname, recons_plot


# Main logic
if not args.just_plot:
    if not (args.psrname and args.parfile and args.timfile):
        print("You must provide the pulsar name, parfile, and timfile for sampling!")
        exit(0)

    # Save command-line arguments
    save_args_to_file(args)

# Set up pulsar and output directories
psrname = args.psrname
print("Pulsar(s) provided:", psrname)
if len(psrname) > 1:
    print("Searching for a common SW signal across multiple pulsars.")

outdir, plotname, recons_plot = setup_outdir(psrname, args.out_dir, args.just_plot)
parfile, timfile = args.parfile, args.timfile

# Initialize libstempo pulsar objects
ltpsr = initialize_pulsars(parfile, timfile, psrname)

# Fit for the timing model
if not args.nofit:
    fit_timing_model(ltpsr, psrname, args.nfit, args.plot_after_fitting)
else:
    print("Skipping timing model fitting.")
    LP.plotres(ltpsr)
    plt.show()

# Create enterprise.Pulsar objects
psrs = create_enterprise_pulsars(ltpsr, psrname)






def hypermodel_prep_func(psrs):
    model_susarla = gp_signals.TimingModel()
    model_susarla += white_noise_block(vary=True)

    model_susarla += red_noise_block(psd='powerlaw', components=30)

    model_susarla += dm_noise_block(psd='powerlaw', components=30)
    
    

    if len(psrname)>1:
        tspan=model_utils.get_tspan(psrs)
    else:
        tspan=psrs.toas.max()-psrs.toas.min()
    model_susarla += common_solar_wind(ACE_prior=False, include_swgp=True,swgp_prior=args.swgp_basis, swgp_basis=args.swgp_basis,Tspan=tspan)

    model_susarla_pta = model_susarla(psrs)

    pta_susarla = signal_base.PTA([model_susarla_pta])


    model_nitu = gp_signals.TimingModel()

    model_nitu += white_noise_block(vary=True)

    model_nitu += red_noise_block(psd='powerlaw', components=30)

    model_nitu += dm_noise_block(psd='powerlaw', components=30)

    psr = Pulsar(ltpsr,drop_t2pulsar=False)

    sw_model_nitu = setup_model_nitu(psr,parfile[0])


    model_nitu += sw_model_nitu

    model_nitu += common_solar_wind(ACE_prior=False, include_swgp=False,Tspan=tspan)


    model_nitu_pta = model_nitu(psrs)

    pta_nitu = signal_base.PTA([model_nitu_pta])



    return model_susarla, model_nitu



if args.hypermodel:
    print("You have chosen the hypermodel")
    outdir += "hypermodel_susarla_nitu"
    plotname += "hypermodel_susarla_nitu"
    recons_plot += "hypermodel_susarla_nitu"
    
    model_susarla, model_nitu = hypermodel_prep_func(psrs)
    
    s = [[model_susarla(psrs)],[model_nitu(psrs)]]
    ptadict = dict.fromkeys(np.arange(0, len(s)))
    for i in range(len(s)):
        ptadict[i] = signal_base.PTA(s[i])

    pta = hypermodel.HyperModel(ptadict,log_weights=None)

    for i in pta.param_names:
        print(i)


    # Define number of iteration to draw
    if args.nsample:
        nsamples = args.nsample
    else:
        nsamples = 2e6

    os.system('mkdir '+outdir)

    # Save parameter names
    with open(outdir+"/pars.txt", "w") as fout:
        for pname in pta.param_names:
            fout.write(pname + "\n")

    # Set initial parameter values from a random draw
    #x0 = np.hstack([p.sample() for p in pta.params])
    x0 = pta.initial_sample()

    # The sampler needs the parameter dimension
    ndim = len(x0)
    #print("\nLH(x0) = %.3f"%(pta.get_lnlikelihood(x0)))

    #print("Prior(x0) = %.3f"%(pta.get_lnprior(x0))) 
    # Define an inital covariance matrix as 'non-informative'. It will be used to set 'jump proposals', used to choose which paramter to update at each iteration
    cov = np.diag(np.ones(ndim) * 1.0**2)



    # Set sampler
    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=outdir)



    # Set Jump Proposals
    jp = JumpProposal(pta, snames=pta.snames)

    sampler.addProposalToCycle(jp.draw_from_prior, 5)

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
                "egp":30}

    for s in sel_sig:
        if any([s in p for p in pta.param_names]):
            sampler.addProposalToCycle(jp.draw_from_par_prior(s), 10)

    sampler.addProposalToCycle(pta.draw_from_nmodel_prior, 25)
    

    # Sample parameters
    if args.no_sample:
        print("You have chosen not to sample the parameters. Finishing the script!")
        exit(0)
    else:
        sampler.sample(x0, int(nsamples), SCAMweight=40, DEweight=60, AMweight=100)
        print("Sampling done")
        exit(0)




def setup_noise_model(args, psrs, parfile, timfile, psrname, outdir, plotname, recons_plot):
    """Sets up noise models and returns the signal model."""
    
    def update_filenames(suffix):
        nonlocal outdir, plotname, recons_plot
        outdir += suffix
        plotname += suffix
        recons_plot += suffix

    ### Timing model marginalization
    s = gp_signals.TimingModel()

    ### White Noise
    if args.wn:
        backend = selections.Selection(selections.no_selection) if args.sims else selections.Selection(selections.by_backend)
        efac = parameter.Uniform(0.1, 5.0)
        equad = parameter.Uniform(-8, -2)
        white_noise = white_signals.MeasurementNoise(efac=efac, selection=backend)
        white_noise += white_signals.TNEquadNoise(log10_tnequad=equad, selection=backend)
        s += white_noise
        update_filenames("_wn")
    else:
        s += white_signals.MeasurementNoise(efac=1, selection=selections.Selection(selections.no_selection))

    ### Red Noise
    if args.rn:
        log10_A = parameter.Uniform(-20, -8)
        gamma = parameter.Uniform(0, 7)
        pl = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
        red_noise = gp_signals.FourierBasisGP(pl, components=args.rn_nbins, name='red_noise')
        s += red_noise
        update_filenames("_rn")

    ### Solar Wind Sigma
    if args.swsigma:
        print("SW sigma is being used")
        psr = Pulsar(parfile[0], timfile[0], drop_t2pulsar=False)
        sw_model_nitu = setup_model_nitu(psr, parfile[0])
        s += sw_model_nitu
        update_filenames("_swsigma")

    ### Solar Wind GP and NESW
    if args.swgp and args.nesw:
        tspan = model_utils.get_tspan(psrs) if len(psrname) > 1 else psrs.toas.max() - psrs.toas.min()
        print("SWGP and NESW are being used")
        freqs = np.linspace(1/tspan, args.swgp_nbins/tspan, args.swgp_nbins)
        if not args.nocutoff:
            freqs = freqs[1/freqs > args.cutoff * yr_in_sec]
        s += common_solar_wind(ACE_prior=False, include_swgp=True, swgp_prior=args.swgp_basis, 
                               swgp_basis=args.swgp_basis, Tspan=tspan, include_n_earth=True)
        update_filenames(f"_nesw_swgp_{len(freqs)}bins")
    
    elif args.swgp:
        tspan = model_utils.get_tspan(psrs) if len(psrname) > 1 else psrs.toas.max() - psrs.toas.min()
        print("SWGP is being used")
        freqs = np.linspace(1/tspan, args.swgp_nbins/tspan, args.swgp_nbins)
        if not args.nocutoff:
            freqs = freqs[1/freqs > args.cutoff * yr_in_sec]
        s += common_solar_wind(ACE_prior=False, include_swgp=True, swgp_prior=args.swgp_basis, 
                               swgp_basis=args.swgp_basis, Tspan=tspan, include_n_earth=False)
        update_filenames(f"_swgp_{len(freqs)}bins")
        
    elif args.nesw:
        tspan = model_utils.get_tspan(psrs) if len(psrname) > 1 else psrs.toas.max() - psrs.toas.min()
        s += common_solar_wind(ACE_prior=False, include_swgp=False, Tspan=tspan)
        update_filenames("_nesw")

    ### DM Variations
    if args.dmgp:
        log10_A = parameter.Uniform(-18, -4)
        gamma = parameter.Uniform(0, 7)
        if args.dmgp_sample_nbins:
            k_dropbin = parameter.Uniform(10, 151)
            pl = drop.dropout_powerlaw(log10_A=log10_A, gamma=gamma, k_drop=None, k_dropbin=k_dropbin)
        else:
            pl = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
        
        dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=args.dmgp_nbins)
        dmgp = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')
        s += dmgp
        update_filenames("_dmgp")

    return s, outdir, plotname, recons_plot


def create_pta_model(psrs, s, psrname):
    """Creates PTA object with the provided signal model."""
    if len(psrname) > 1:
        model = [s(psr) for psr in psrs]
    else:
        model = [s(psrs)]
    
    pta = signal_base.PTA(model)
    params = {"%s_efac" % psrname: 1.}
    pta.set_default_params(params)
    return pta


# Main function to set up noise model and PTA
s, outdir, plotname, recons_plot = setup_noise_model(args, psrs, parfile, timfile, psrname, outdir, plotname, recons_plot)
pta = create_pta_model(psrs, s, psrname)






if not args.just_plot:
    # Check LH and prior values
    x0 = np.hstack([p.sample() for p in pta.params])
    print("Parameters and priors:\n")
    for i, p in enumerate(pta.param_names):
        print(f"{p:<37s} : {x0[i]:6.3f}")
    print(f"\nLH(x0) = {pta.get_lnlikelihood(x0):.3f}")
    print(f"Prior(x0) = {pta.get_lnprior(x0):.3f}")

    # Set sampler if sampling is required
    if not args.no_sample:
        resume = False
        if not os.path.isdir(outdir):
            subprocess.check_output(f'mkdir -p {outdir}', shell=True)
        
        Nsamples = int(args.nsample)
        np.savetxt(f'{outdir}/pars.txt', pta.param_names, fmt='%s')
        os.system(f'mv command_line_args.txt {outdir}')

        # Initialize sampler
        cov = np.diag(np.ones(len(x0)) * 0.01**2)
        sampler = ptmcmc(len(x0), pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=outdir, resume=resume)
        jp = JumpProposal(pta)

        # Set jump proposals
        sel_sig = {"timing_model": 20, "rn_": 10, "red_noise_": 10, "dm_": 10, "fcn_": 10, 
                   "cn_": 10, "expd": 10, "yearly": 10, "crs": 30, "mass": 10, "orb_elements": 30,
                   "frame_drift_rate": 30, "egp": 30, "gp_sw": 10, "n_earth": 10}
        
        for s, n in sel_sig.items():
            if any(s in p for p in pta.param_names):
                sampler.addProposalToCycle(jp.draw_from_par_prior(s), n)

        # Launch sampling
        sampler.sample(x0, Nsamples, SCAMweight=40, AMweight=15, DEweight=60)
        print({par.name: par.sample() for par in pta.params})
    else:
        print("Sampling skipped.")
        exit(0)

if args.just_plot:
    if not args.chain_dir:
        print("Chain directory is required for plotting!")
        exit(0)
    print(f"Plotting posteriors using chain directory: {args.chain_dir}")
    plotting(outdir=args.chain_dir, pta=pta, psr=psrs, plotname=plotname, recons_plot=recons_plot)
    print("Script finished!")
    exit(0)

if not args.no_plot:
    print("Plotting posteriors and time-domain reconstruction")
    plotting(outdir=outdir, pta=pta, psr=psrs, plotname=plotname, recons_plot=recons_plot)
else:
    print("Plotting skipped.")
    exit(0)