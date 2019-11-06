# -*- coding: iso-8859-1 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
#from hammersley import hammersley
from chaospy import sequences 

def icdf_transform_1d(su, x, pdfx):
    # sur is sample from a uniform distribution [0,1]
    # x is the coordinates of the transformed distribution
    # pdf is the pdf of x
    cdf = np.cumsum(pdfx); cdf /= np.max(cdf)
    return np.interp(su, cdf, x)
    
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

# these numbers are all SI base units
def make_beam(npart=int(1e6), tay13scalefactor=1, power_profile_file='gaus_flat_triangle.npy', plotQ=False):
    # tay13scalefactor <float> should be between 0 and 1.25; 0 => Gaussian & 1 => flat-top

    # create beam
    beam = sequences.create_hammersley_samples(order=npart, dim=1).T

    # load temporal profile (assuming current = constant * power)
    pvst = interpolate_profile(profilefile=power_profile_file, tay13scalefactor=tay13scalefactor)
    #d=pd.read_csv('dist100.part',delim_whitespace=True) # reminder for handling variable space delimiters
    t = pvst[:,0]*1e-12; dt = np.abs(t[1]-t[0])# seconds
    p = pvst[:,1] # power (arb. units)

    # apply temporal profile
    beam[:,0] = icdf_transform_1d(beam[:,0],t,p)
    ts = beam[:,0]

    # how does the temporal profile compare to the input distribution?
    if plotQ:
        
        binfills, binedges, patches = plt.hist(ts,500);
        arearatio = np.sum(binfills)*np.abs(binedges[1]-binedges[0])/np.sum(p)/dt
        plt.scatter(t, p*arearatio,s=1,c='Red'); 
        plt.plot(t, p*arearatio); 
        plt.xlabel('time (s)'); plt.ylabel('number of particles')
        plt.show(); plt.close()

    return beam
