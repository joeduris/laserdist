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
    
def interpolate_profile(profilefile='gaus_flat_triangle.npy', tay13scalefactor=0, nradius=3):
    # profilefile <string> path to profile data to interpolate
    # tay13scalefactor <float> should be between 0 and 1.25; 0 => Gaussian & 1 => flat-top
    # nradius <int> number of grid points away to grab data for interpolation
    
    sf = tay13scalefactor # shorthand
    
    if nradius < 1:
        print('Ah! Ah! Aah! You shouldn\'t interpolate without points.')
        nradius = 3
    
    # load simulated profile data
    #ps = np.genfromtxt(profilefile, delimiter=',')[1:] # first row = column descriptions
    ps = np.load(profilefile)
    
    # find the unique coords and bounds on inputs
    unique_ts = np.unique(ps[:,1])
    unique_fs = np.unique(ps[:,0])
    minf = min(unique_fs); maxf = max(unique_fs)

    # keep within bounds
    sf = min(sf, maxf); sf = max(sf, minf)

    # select nearby tay13scalefactors
    sf_iloc = np.interp(sf, unique_fs, np.arange(len(unique_fs)))
    cut = np.abs(np.arange(len(unique_fs))-sf_iloc) < nradius
    
    # select profiles for these nearby points
    bigcut = np.sum([ps[:,0] == f for f in unique_fs[cut]],axis=0) == 1
    pss = ps[bigcut]
    
    # interpolate on the selected profiles
    myinterp = []
    for t in unique_ts:
        cut = pss[:,1] == t
        myinterp += [[t,np.interp(sf,pss[cut,0],pss[cut,2])]]
    myinterp = np.array(myinterp,ndmin=2)
    
    return myinterp
    
def quartic_gaussian(x, xfwhm):
    return 2.**(-(2.*x/xfwhm)**4)

def filtered_green_profile(tay13scalefactor=1., filter_bw_fwhm_nm=1., filter_bg_passfraction=0., power_profile_file='gaus_flat_triangle.npy', plotQ=False):
    # tay13scalefactor <float> should be a real number in range [-3.,3.]:
    #   NOTE: the sign of tay13scalefactor flips the origin of time
    #   0 => Gaussian; 1 => flat-top; 2 => triangle; 3 => smoother triangle facing opposite direction
    # filter_bw_fwhm_nm <float> [0.,1e3] is the fwhm width of the band pass filter; 1 should smooth most ripples; 10 should pass everything
    # filter_bg_passfraction <float> is the fraction of power [0.,1.] to pass independent of frequency; default is 0, but 1 is equivalent to no bandpass filter

    # validate range of t_sign
    t_sign = 1.*np.sign(tay13scalefactor);
    if np.abs(t_sign) < 1.:
        t_sign = 1.
    
    # load temporal profile (assuming current = constant * power)
    pvst = interpolate_profile(profilefile=power_profile_file, tay13scalefactor=np.abs(tay13scalefactor))
    pvst[:,0] *= t_sign # reverse time
    t = pvst[:,0]*1e-12; dt = np.abs(t[1]-t[0])# seconds
    p = pvst[:,1] # power (arb. units)
    
    lambda0 = 515.e-9 # green wavelength in m (freq. doubled 1030 nm)
    f0 = 2.998e8/lambda0 # green frequency in Hz (doubled 1030 nm)
    Df=1/dt # width of frequency window
    df=1/(max(t)-min(t)) # delta frequency steps
    Df=df*(len(t)-1) # once again, frequency window
    dlambda_nm = 2.998e8/f0**2*df*1e9 # delta wavelength steps in picometers
    fft_filter_bw = filter_bw_fwhm_nm / dlambda_nm # delta frequency steps in units of df
    
    fft = np.fft.fft(p**0.5) # fft of the field strength (assumes slowly varying phase)
    
    fft_index = np.fft.ifftshift(np.arange(len(fft))-len(fft)/2.+0.5) # zero at peak of field
    bgfrac = np.max([0,np.min([1.,filter_bg_passfraction])]);
    if not (bgfrac - filter_bg_passfraction)==0:
        print('WARNING: filter_bg_passfraction =', filter_bg_passfraction, 'is not valid so changing to closest valid value in range [0,1]:', bgfrac)
    fft_filter = (1.-bgfrac)*quartic_gaussian(fft_index,fft_filter_bw) + bgfrac
    if plotQ:
        lambdas = 1e9*(lambda0+np.fft.fftshift(fft_index)*dlambda_nm)
        absfft = np.fft.fftshift(np.abs(fft)); absfft /= np.max(absfft)
        shiftfilt = np.fft.fftshift(fft_filter)
        plt.plot(lambdas, absfft, label='input'); plt.plot(lambdas, shiftfilt, label='filter')
        plt.plot(lambdas, absfft*shiftfilt, label='output'); plt.xlabel('wave length (nm)'); 
        plt.legend(); plt.show(); plt.close()
    
    p = np.abs(np.fft.ifft(fft * fft_filter))**2
    if plotQ:
        plt.plot(pvst[:,0],pvst[:,1], label='input')
        plt.plot(pvst[:,0],p, label='output'); plt.xlabel('time (ps)'); 
        plt.legend(); plt.show(); plt.close()
    pvst[:,1] = p
    
    return pvst

# these numbers are all SI base units
def make_beam(npart=int(5e4), tay13scalefactor=1., filter_bw_fwhm_nm=1., filter_bg_passfraction=0., t_origin_ps=0., power_profile_file='gaus_flat_triangle.npy', sigmax=300e-6, cut_radius_x=450e-6, pr_eV_mean=4.*1240./1030.-2.86, pr_eV_rms=25.7e-3, sigmagamma=0.0005/511., plotQ=False):
    # tay13scalefactor <float> should be a real number in range [-3.,3.]:
    #   NOTE: the sign of tay13scalefactor flips the origin of time
    #   0 => Gaussian; 1 => flat-top; 2 => triangle; 3 => smoother triangle facing opposite direction
    # filter_bw_fwhm_nm <float> [0.,1e3] is the fwhm width of the band pass filter; 1 should smooth most ripples; 10 should pass everything
    # filter_bg_passfraction <float> is the fraction of power [0.,1.] to pass independent of frequency; default is 0, but 1 is equivalent to no bandpass filter
    # t_origin_ps <float> is the time in ps to shift the beam by (default is 0.)
    
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
    #pvst = interpolate_profile(profilefile=power_profile_file, tay13scalefactor=tay13scalefactor)
    pvst = filtered_green_profile(tay13scalefactor=tay13scalefactor, filter_bw_fwhm_nm=filter_bw_fwhm_nm, filter_bg_passfraction=filter_bg_passfraction, power_profile_file=power_profile_file, plotQ=plotQ)
    #d=pd.read_csv('dist100.part',delim_whitespace=True) # reminder for handling variable space delimiters
    t = (pvst[:,0]-t_origin_ps)*1e-12; dt = np.abs(t[1]-t[0])# seconds
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
        cnames = ['x (m)','y (m)','time (s)','px (eV/c)','py (eV/c)','pz (eV/c)']
        for p in range(1):
            plt.hist2d(beam[:,2*p],beam[:,2*p+1],100)
            plt.xlabel('plane '+str(2*p)+': '+cnames[2*p]); plt.ylabel('plane '+str(2*p+1)+': '+cnames[2*p+1])
            plt.show(); plt.close()
            
        plt.hist(beam[:,0],100,label='x')
        plt.hist(beam[:,1],100,label='y')
        plt.xlabel('position (m)'); plt.legend(); plt.show()
            
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        
        # momenta distribution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = plt.axes(projection='3d')
        nplotmax3d = 10000
        if npart > nplotmax3d:
            print('INFO: truncating number of particles plot for sanity.')
        ax.scatter(beam[:nplotmax3d,3], beam[:nplotmax3d,4], beam[:nplotmax3d,5], c=beam[:nplotmax3d,5], alpha=0.33, cmap='viridis', linewidth=0.5);
        ax.set_xlabel('$p_x$ (eV/c)');ax.set_xlabel('$p_y$ (eV/c)');ax.set_xlabel('$p_z$ (eV/c)');plt.show()
        
        plt.hist(np.sqrt(beam[:nplotmax3d,3]**2 + beam[:nplotmax3d,4]**2 + beam[:nplotmax3d,5]**2), 51); 
        plt.xlabel('radial momenta (eV/c)'); plt.show()

    return beam
