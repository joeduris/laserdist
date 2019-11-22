# -*- coding: iso-8859-1 -*-

# Joe Duris - jduris@slac.stanford.edu

import numpy as np
from scipy.special import erfinv, erf
#from hammersley import hammersley
from chaospy import sequences 

def icdf_transform_1d(su, x, pdfx):
    # sur is sample from a uniform distribution [0,1]
    # x is the coordinates of the transformed distribution
    # pdf is the pdf of x
    cdf = np.cumsum(pdfx); cdf /= np.max(cdf)
    return np.interp(su, cdf, x)
    
def normal_from_uniform(su,means=None,sigmas=None):
    # su is sample from a uniform distribution [0,1]
    # output is normally distributed
    oneD = False
    if len(np.shape(su)) == 1:
        oneD = True
        su = np.array(su,ndmin=2).T
    npts, dim = np.shape(su)
    if means is None:
        means = np.zeros(dim)
    if sigmas is None:
        sigmas = np.ones(dim)
    xs = np.sqrt(2)*erfinv(-1+2*np.array(su,ndmin=2))
    xs = np.transpose(np.array(sigmas,ndmin=2).T * xs.T) + np.array(means)
    if oneD:
        xs = xs[:,0]
    return xs

def cut_normal_2dradial_from_uniform(su,cut_radius_nsigma=10):
    # su is sample from a uniform distribution [0,1]
    # output ~ x*exp(-x^2/2)
    oneD = False
    if len(np.shape(su)) == 1:
        oneD = True
        su = np.array(su,ndmin=2).T
    npts, dim = np.shape(su)
    a2 = cut_radius_nsigma**2
    rs = np.sqrt(a2 - 2 * np.log(su + np.exp(a2/2.) - su * np.exp(a2/2.)))
    if oneD:
        rs = rs[:,0]
    return rs
    
def cut_normal_round_from_uniform_2d(su, gauss_sigma, cut_radius):
    # su is samples from a uniform distribution with shape (npts, dim=2)
    # note: stdev in x and in y of output distribution is gauss_sigma*np.sqrt(1 + cut_radius**2/(2 - 2*np.exp(cut_radius**2/2)))
    rs = gauss_sigma * cut_normal_2dradial_from_uniform(su[:,0],cut_radius_nsigma=cut_radius/gauss_sigma)
    phis = 2.*np.pi*su[:,1]
    nd = np.zeros_like(su)
    nd[:,0] = rs * np.cos(phis)
    nd[:,1] = rs * np.sin(phis)
    return nd
    
# numerically unstable as hell for r0/sigmar >> 15, but by then a normal Gaussian isn't a bad approx
def cdf_r_spherical(r,r0,sigmar):
    s = np.shape(r)
    r = np.array(r,ndmin=1)
    cut = r == r0
    cdf_r0 = ((-2 + np.exp(-r0**2/(2*sigmar**2)))*r0*sigmar**2 + np.sqrt(np.pi/2)*sigmar*(r0**2 + sigmar**2)* erf(r0/(np.sqrt(2)*sigmar)))/((r0*sigmar**2)/np.exp(r0**2/(2*sigmar**2)) + np.sqrt(np.pi/2)*sigmar*(r0**2 + sigmar**2)*(1 + erf(r0/(np.sqrt(2)*sigmar))))
    cdf = (-(((-(np.exp(r**2/(2*sigmar**2))*r0) + np.exp((r*r0)/sigmar**2)*(r + r0))*sigmar**2)/ np.exp((r**2 + r0**2)/(2*sigmar**2))) + (np.sqrt(np.pi/2)*(r - r0)*sigmar*(r0**2 + sigmar**2)* erf(np.sqrt((r - r0)**2)/(np.sqrt(2)*sigmar)))/np.sqrt((r - r0)**2) +  np.sqrt(np.pi/2)*sigmar*(r0**2 + sigmar**2)*erf(r0/(np.sqrt(2)*sigmar))) /     ((r0*sigmar**2)/np.exp(r0**2/(2*sigmar**2)) + np.sqrt(np.pi/2)*sigmar*(r0**2 + sigmar**2)* (1 + erf(r0/(np.sqrt(2)*sigmar))))
    cdf[cut] = cdf_r0
    cdf = np.reshape(cdf, s)
    
    return cdf
    
def invcdf_r_spherical(su, r0, sigmar, rmax=None, steps=1001):
    # su is samples from a uniform distribution with shape (npts, dim=1)
    if rmax is None:
        rmax = r0 + 7. * sigmar
    try:
        npts, dim = np.shape(su)
        if dim > 1:
            print('WARNING: passing too many dimensions!')
        su = su[:,0]
    except:
        pass
    # generate the cdf for inversion
    #[r,cdf] = np.transpose([[r,-((np.sqrt(2./np.pi)*r)/np.exp(r**2/2.)) + erf(r/np.sqrt(2))] for r in np.linspace(0,rmax,steps)])
    fallback = False; numstablethresh = 15
    if np.abs(r0/sigmar) < numstablethresh:
        [r,cdf] = np.transpose([[r,cdf_r_spherical(nld(r),nld(r0),nld(sigmar))] for r in np.linspace(0,rmax,steps)])
        if np.any(np.isnan(cdf)):
            fallback = True
        else:
            su[su > cdf[-2]] = cdf[-2] # map the rest to rmax to avoid extrapolation
            rs = np.interp(su,cdf,r)
    if np.abs(r0/sigmar) >= numstablethresh or fallback:
        rs = normal_from_uniform(su) * sigmar + r0
    
    #rs = np.reshape(rs,(len(su),1))
    
    return rs
    
def invcdf_theta_spherical(su,theta_range=[0,np.pi/2]):
    # su is samples from a uniform distribution with shape (npts, dim=1)
    # key points
    [c0,c1] = (1-np.cos(theta_range))/2
    thetas = 2 * np.arcsin(np.sqrt(np.min([c0,c1])+np.abs(c0-c1)*su))
    
    return thetas
    
def zcut_spherical_shell_normal_radial_from_uniform_3d(su, rmean, rstd, phi_range=[0,2*np.pi], theta_range=[0,np.pi/2]):
    # su is samples from a uniform distribution with shape (npts, dim=3)
    # rmean, rstd, are the radial properties
    # phi_range is the range of azimuthal angles to (uniformly) cover
    # theta_range is the range of polar angles to (uniformly) cover
    
    rs = invcdf_r_spherical(su[:,0], rmean, rstd)
    phis = np.abs(np.diff(phi_range))[0]*su[:,1]+np.min(phi_range)
    thetas = invcdf_theta_spherical(su[:,2],theta_range=theta_range)
    nd = np.zeros_like(su)
    nd[:,0] = rs * np.sin(thetas) * np.cos(phis)
    nd[:,1] = rs * np.sin(thetas) * np.sin(phis)
    nd[:,2] = rs * np.cos(thetas)
    return nd
    
def custom_power_profile_ps(t_width_ps=20., t_origin_ps=0., dt_low_ps=1., dt_high_ps=1., slope=0., t_range_ps=[None,[-80,80]][0], dt_range_ps=0.01, force_t_rms_equal_t_width_ps=False, force_t_fwhm_equal_t_width_ps=False, force_t_mean_equal_t_origin_ps=False, force_t_peak_equal_t_origin_ps=False):
    # units are in picoseconds except for slope
    # if t_range_ps is a 2-entry array, it defines the sampled time range; otherwise, this is estimated from input params
    # force_t_rms_equal_t_width_ps <bool> rescales time coords so that profile rms duration is equal to t_width_ps
    # force_t_fwhm_equal_t_width_ps <bool> rescales time coords so that profile fwhm duration is equal to t_width_ps
    # force_t_mean_equal_t_origin_ps <bool> shifts time coords so that profile mean is equal to t_origin_ps
    # force_t_peak_equal_t_origin_ps <bool> shifts time coords so that profile peak is equal to t_origin_ps
    
    t_width_ps = np.abs(t_width_ps)
    t1 = t_origin_ps - t_width_ps; t2 = t_origin_ps + t_width_ps
    dt1 = np.abs(dt_low_ps); dt2 = np.abs(dt_high_ps); dt = np.abs(t2 - t1)
    
    if t_range_ps is None:
        t_range_ps = 2. * np.array([t1 - dt1, t2 - dt2])
    
    npts = np.abs(np.diff(t_range_ps) / dt_range_ps)
    t = np.linspace(np.min(t_range_ps), np.max(t_range_ps), npts)
    base = 3. + 2.*np.sqrt(2)
    
    p = 2. / (1. + np.exp(-8.*t*slope/dt))
    p /= 1. + base**(-2.*(t - t1)/dt1)
    p /= 1. + base**(2.*(t - t2)/dt2)
    
    if force_t_rms_equal_t_width_ps:
        tmean = np.dot(t,p) / np.sum(p)
        trms = np.sqrt(np.dot(t**2,p) / np.sum(p) - tmean)
        t *= trms / t_width_ps
        
    if force_t_fwhm_equal_t_width_ps:
        preduced = np.abs(p/np.max(p) - 0.5)
        ipeak = p.argmax()
        ilow = preduced[:ipeak].argmin()
        ihigh = preduced[ipeak:].argmin()
        tfwhm = t[ihigh] - t[ilow]
        t *= tfwhm / t_width_ps
        
    if force_t_mean_equal_t_origin_ps:
        tmean = np.dot(t,p) / np.sum(p)
        t -= tmean
        
    if force_t_peak_equal_t_origin_ps:
        ipeak = p.argmax()
        t -= t[ipeak]
        
    pvst = np.vstack((t,p)).T
    return pvst

def make_beam(npart=int(5e4), t_width_ps=20., t_origin_ps=0., dt_low_ps=1., dt_high_ps=1., slope=0., force_t_rms_equal_t_width_ps=False, force_t_fwhm_equal_t_width_ps=False, force_t_mean_equal_t_origin_ps=False, force_t_peak_equal_t_origin_ps=False, sigmax=300e-6, cut_radius_x=450e-6, pr_eV_mean=4.*1240./1030.-2.86, pr_eV_rms=25.7e-3, plotQ=False):
    # t_width_ps <float> is the pulse duration in ps; if slope==0, then this is the FWHM width; otherwise, this is an upper bound on the width
    # t_origin_ps <float> is the time in ps to shift the beam by
    # dt_low_ps <float> is the duration of the lower ramp
    # dt_high_ps <float> is the duration of the upper ramp
    # slope <float> sets the normalized slope (roughly relative change over half width) of the current in the core of the beam: 
    #               keep between -1 and 1 (although feel free to play with it)
    # force_t_rms_equal_t_width_ps <bool> rescales time coords so that profile rms duration is equal to t_width_ps
    # force_t_fwhm_equal_t_width_ps <bool> rescales time coords so that profile fwhm duration is equal to t_width_ps
    # force_t_mean_equal_t_origin_ps <bool> shifts time coords so that profile mean is equal to t_origin_ps
    # force_t_peak_equal_t_origin_ps <bool> shifts time coords so that profile peak is equal to t_origin_ps
    
    # pr_eV_mean is energy above ionization in eV: 4.*1240./1030. eV is enery of UV and 2.86 eV is the workfunction of the Cesium Telluride
    # pr_eV_rms is the standard deviation of the radial momenta in eV: 25.7 meV is kT at room temp, although this should probably be dominated by the QE(photon energy) curve response?
    # interestingly, the QE should go down with time as we hit the cathode with UV (if intense enough)
    # said another way, the work function increases with UV exposure => excess momentum and emittance should decrease with increased operation
    # https://indico.classe.cornell.edu/event/15/contributions/394/attachments/290/364/harkayYusof-p3-2012-final.pdf
    
    # NOTE: we'll lay out the beam as a numpy array with shape (npart, 6)
    #       where the dimensions are ordered as x, y, time, px, py, pz
    #       ASTRA takes z as zero

    # create beam
    #beam = np.random.rand(npart,6)
    beam = sequences.create_hammersley_samples(order=npart, dim=6).T
    
    # figure out spread in angles 

    # make the non-time profiles
    beam[:,:2] = cut_normal_round_from_uniform_2d(beam[:,:2], sigmax, cut_radius_x)
    beam[:,3:] = zcut_spherical_shell_normal_radial_from_uniform_3d(beam[:,3:], pr_eV_mean, pr_eV_rms, phi_range=[0,2*np.pi], theta_range=[0,np.pi/2])
    
    # load temporal profile (assuming current = constant * power)
    pvst = custom_power_profile_ps(t_width_ps=t_width_ps, t_origin_ps=t_origin_ps, dt_low_ps=dt_low_ps, dt_high_ps=dt_high_ps, slope=slope, t_range_ps=None, dt_range_ps=0.01, force_t_rms_equal_t_width_ps=force_t_rms_equal_t_width_ps, force_t_fwhm_equal_t_width_ps=force_t_fwhm_equal_t_width_ps, force_t_mean_equal_t_origin_ps=force_t_mean_equal_t_origin_ps, force_t_peak_equal_t_origin_ps=force_t_peak_equal_t_origin_ps)
    t = pvst[:,0]*1e-12; dt = np.abs(t[1]-t[0])# seconds
    p = pvst[:,1]**2 # power (arb. units) --- loaded power is green; simulation shows that SHG in the next crystal (green -> UV) just squares the input power profile

    # apply temporal profile
    beam[:,2] = icdf_transform_1d(beam[:,2],t,p)

    # how does the temporal profile compare to the input distribution?
    if plotQ:
        
        import matplotlib.pyplot as plt
        
        # temporal distribution
        binfills, binedges, patches = plt.hist(beam[:,2],500); plt.close()
        arearatio = np.sum(binfills)*np.abs(binedges[1]-binedges[0])/np.sum(p)/dt
        plt.plot(t, p*arearatio, linewidth=0.8); 
        plt.scatter(t, p*arearatio,s=0.05,c='Red');
        plt.hist(beam[:,2],500);
        plt.xlabel('time (s)'); plt.ylabel('number of particles')
        plt.show(); plt.close()
        
        # transverse position distribution
        cnames = ['x (um)','y (um)','time (s)','px (eV/c)','py (eV/c)','pz (eV/c)']
        unitscales = [1e6,1e6,1,1,1,1]
        for p in range(1):
            plt.hist2d(beam[:,2*p]*unitscales[2*p],beam[:,2*p+1]*unitscales[2*p+1],100)
            plt.xlabel('plane '+str(2*p)+': '+cnames[2*p]); plt.ylabel('plane '+str(2*p+1)+': '+cnames[2*p+1])
            plt.show(); plt.close()
            
        plt.hist(beam[:,0]*unitscales[0],100,label='x')
        plt.hist(beam[:,1]*unitscales[1],100,label='y')
        plt.xlabel('position (m)'); plt.legend(); plt.show()
            
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        
        # momenta distribution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = plt.axes(projection='3d')
        nplotmax3d = 50000
        if npart > nplotmax3d:
            print('INFO: truncating number of particles plot to',nplotmax3d,'for sanity.')
        ax.scatter(beam[:nplotmax3d,3], beam[:nplotmax3d,4], beam[:nplotmax3d,5], c=beam[:nplotmax3d,5], alpha=0.33, cmap='viridis', linewidth=0.5);
        ax.set_xlabel('$p_x$ (eV/c)');ax.set_ylabel('$p_y$ (eV/c)');ax.set_zlabel('$p_z$ (eV/c)');plt.show()
        
        plt.hist(np.sqrt(beam[:nplotmax3d,3]**2 + beam[:nplotmax3d,4]**2 + beam[:nplotmax3d,5]**2), 101); 
        plt.xlabel('radial momenta (eV/c)'); plt.show()

    return beam
