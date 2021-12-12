'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski'
'''
import numpy as np
import scipy.special as special
import math

import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import CAMB_General_Code 
import j0j0
from mcfit import P2xi
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from fastpt import *

class Info:
    def __init__(self, alpha_):
        self.alpha = alpha_
        import fastpt
        import fastpt.HT as HT

        # import the Core Cosmology Library (CCL) if you have it
        try:
            import pyccl as ccl
            have_ccl = True
        except:
            cl = False
        have_ccl = False
        # If you want to test HT against external Bessel transform code, e.g. mcfit
           
        try:
            from mcfit import P2xi
            have_mcfit = True
        except:
            have_mcfit = False
  
        ## Get from CCL (which runs CLASS by default)
        if have_ccl:
            # set two cosmologies
            cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
            cosmo2 = ccl.Cosmology(Omega_c=0.30, Omega_b=0.045, h=0.67, A_s=2.0e-9, n_s=0.96)
  
            # Get the linear power spectrum at z=0 for our given cosmologies
            # k array to be used for power spectra
            nk = 512
            log10kmin = -5
            log10kmax = 2
            ks = np.logspace(log10kmin,log10kmax,nk)
            pk_lin_z0 = ccl.linear_matter_power(cosmo,ks,1)
            pk_lin_z0_2 = ccl.linear_matter_power(cosmo2,ks,1)
    
        ## Or get from pre-computed CAMB run
        # This file is in the same examples/ folder
        self.d = np.loadtxt('Pk_test.dat')
        self.k = self.d[:, 0]
        self.pk = self.d[:, 1]
        self.p22 = self.d[:, 2]
        self.p13 = self.d[:, 3]
  
        #if not have_ccl:
        self.ks = self.k
        self.pk_lin_z0 = self.pk
        self.pk_lin_z0_2 = None
                
        ## Or get from your preferred Boltzmann code
    
        # Note: k needs to be evenly log spaced. FAST-PT will raise an error if it's not.
        # We have an issue to add automatic interpolation, but this is not yet implemented.
    
        # Evaluation time scales as roughly N*logN. Tradeoff between time and accuracy in choosing k resolution.
        # Currently, k sampling must be done outside of FAST-PT. This feature will also be added.
    
        # Set FAST-PT settings.
    
        # the to_do list sets the k-grid quantities needed in initialization (e.g. the relevant gamma functions)
        self.to_do = ['one_loop_dd', 'dd_bias', 'one_loop_cleft_dd', 'IA_all', 'OV', 'kPol', 'RSD', 'IRres']
    
        self.pad_factor = 1 # padding the edges with zeros before Pk repeats
        self.n_pad = self.pad_factor*len(self.ks)
        self.low_extrap = -5 # Extend Plin to this log10 value if necessary (power law)
        self.high_extrap = 3 # Extend Plin to this log10 value if necessary (power law)
        self.P_window = None # Smooth the input power spectrum edges (typically not needed, especially with zero padding)
        self.C_window = .75 # Smooth the Fourier coefficients of Plin to remove high-frequency noise.
    
        # FAST-PT will parse the full to-do list and only calculate each needed quantity once.
        # Ideally, the initialization happens once per likelihood evaluation, or even once per chain.
    
        self.fpt_obj = FASTPT(self.ks,to_do=self.to_do,low_extrap=self.low_extrap,high_extrap=self.high_extrap,n_pad=self.n_pad)
    
        #fpt_obj_temp = fpt.FASTPT(k,to_do=to_do,low_extrap=low_extrap,high_extrap=high_extrap,n_pad=n_pad)
    
        # Parameters for a mock DESI sample at 0.6 < z < 0.8
    
        # First, we need the growth factor and cosmology
        def growth_factor(cc,zz):
            '''Retunrs linear growth factor for vector zz'''
            if isinstance(zz,float):
                zz = np.array([zz])
            afid = 1.0/(1.0+zz)
            if isinstance(afid,float):
                afid = np.array([afid])
            zval = 1./np.array(list(map(lambda x: np.logspace(x,0.0,100),np.log10(afid)))).transpose() - 1.0
            #zval = 1/np.logspace(np.log10(afid),0.0,100)-1.0
            Dz   = np.exp(-np.trapz(cc.Om(zval)**0.55,x=np.log(1/(1+zval)),axis=0))
            return(Dz)
                
        # round-number cosmology is good enough!
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70.,Om0=0.3)
    
        #zmin and zmax
        self.zmin = 0.6
        self.zmax = 0.8
        self.zbar = 0.5 * (self.zmin + self.zmax)
    
        # number density
        self.number_density = 6e-4 # from Rongpu's targeting paper, in (h^1 Mpc)^-3
    
        # Growth factor
        self.growth = growth_factor(cosmo, self.zbar)
    
        # Bias from 1611.00036
        self.b1 = 1.7/self.growth
    
        # Higher biases from consistency relations
        # from 3.29 in https://arxiv.org/pdf/1611.09787.pdf
        self.delta_cr = 1.686
        #self.delta_cr = 0.
        self.nu_c = ((self.b1-1) * self.delta_cr + 1)**0.5
        #self.nu_c = 0.
        self.b2 = 8./21. * (self.nu_c**2. - 1)/self.delta_cr + self.nu_c**2./(self.delta_cr**2.) * (self.nu_c**2. - 3.)
        #self.b2 = 0.
    
        # from pg 4 of https://arxiv.org/pdf/2008.05991.pdf
        # Note that they are using bias definition appropriate for fastpt (there is a factor of 32/315 that is absorbed into
        # b3nl versus the Saito paper that they cite)
        self.bs = -4./7. * (self.nu_c**2. - 1)/self.delta_cr
        #self.bs = 0.
        self.b3nl = self.b1 - 1.
        #self.b3nl = 0.
    
        # Comoving volume and effective volume
        self.v_survey = (4./3.)*np.pi * ((cosmo.comoving_distance(self.zmax).value*cosmo.h)**3. - (cosmo.comoving_distance(self.zmin).value*cosmo.h)**3.)
    
        # Need an effective power spectrum
        from scipy.interpolate import InterpolatedUnivariateSpline as Spline
        self.pk_spline = Spline(self.k,self.pk_lin_z0)
        self.keff = 0.14
        self.mueff = 0.6 # these values are from 1611.00036
        self.Omz = 0.3 * (1+self.zbar)**3./(0.3 * (1+self.zbar)**3 + 0.7)
        self.f = self.Omz ** 0.55
        self.P_eff = (self.b1 + self.f * self.mueff**2.) **2. * self.growth **2. * self.pk_spline(self.keff)
        self.effective_volume = (1 + (1./(self.number_density * self.P_eff)))**-2. * self.v_survey
    
        # Monopole or xi(r)?
    
        # For PT, we need to multiply by the relevant powers of the growth factor.
        # For simplicity, we will do this all at z=0, where growth = 1. But we will keep the factors explicit.
        self.g2 = self.growth**2
        self.g4 = self.growth**4
        
        self.P_bias_E = self.fpt_obj.one_loop_dd_bias_b3nl(self.pk_lin_z0, C_window=self.C_window)
    
        # Output individual terms
        self.Pd1d1 = self.g2 * self.pk_lin_z0 + self.g4 * self.P_bias_E[0] # could use halofit or emulator instead of 1-loop SPT
        self.Pd1d2 = self.g4 * self.P_bias_E[2]
        self.Pd2d2 = self.g4 * self.P_bias_E[3]
        self.Pd1s2 = self.g4 * self.P_bias_E[4]
        self.Pd2s2 = self.g4 * self.P_bias_E[5]
        self.Ps2s2 = self.g4 * self.P_bias_E[6]
        self.Pd1p3 = self.g4 * self.P_bias_E[8]
        self.s4 =  self.g4 * self.P_bias_E[7] # sigma^4 which determines the (non-physical) low-k contributions
        
        self.P_IRres = self.g2 * self.fpt_obj.IRres(self.pk_lin_z0, C_window=self.C_window)
        # Note that this function needs documentation/validation
    
        self.r, self.xi_IRres = HT.k_to_r(self.ks, self.P_IRres,1.5,-1.5,.5, (2.*np.pi)**(-1.5))
    
        # Combine for P_gg or P_mg
        self.P_gg = ((self.b1 * self.b1) * self.P_IRres) #+
                    #0.5*(self.b1 * self.b2 * 2) * self.Pd1d2 +
                    #0.25*(self.b2 * self.b2) * (self.Pd2d2 - 2.*self.s4) +
                    #0.5*(self.b1 * self.bs * 2) * self.Pd1s2 +
                    #0.25*(self.b2 * self.bs * 2) * (self.Pd2s2 - (4./3.)*self.s4) +
                    #0.25*(self.bs * self.bs) * (self.Ps2s2 - (8./9.)*self.s4) +
                    #0.5*(self.b1 * self.b3nl * 2) * self.Pd1p3)
    
        self.x = [_ for _ in range(len(self.P_gg))]
        
        #Recalculating the CF
        self.r_bins = np.linspace(30, 180, 31)
        self.r = 0.5 * (self.r_bins[1:] + self.r_bins[:-1])
        self.xi_gg = np.zeros(len(self.r))
        self.R1, self.R2 = np.meshgrid(self.r, self.r)
        self.delta_r = self.r_bins[1] - self.r_bins[0] 
            
    def get_r(self):
        return self.r
    
    
    def calc_covariance_matrix(self):
        if os.path.exists('covariance_matrix_11-26-21.txt'):
            self.covariance_matrix = np.loadtxt('covariance_matrix_11-26-21.txt', usecols=range(30))
        else:
            #New Method for Covariance Matrix
            for i in range(len(self.r)):
                self.xi_gg[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * \
                                       special.spherical_jn(0, self.alpha * self.ks * self.r_bins[i]) * \
                                           np.exp(-self.ks**2) * self.P_gg * np.gradient(self.ks))
              
            #r_bins = np.linspace(r_min, r_max, 100)
            self.covariance_matrix = np.zeros((len(self.r), len(self.r)))
            
            for i in range(len(self.r)):
                print(i)
                N_small_r = 25
                r_bins_1 = np.linspace(self.r_bins[i], self.r_bins[i+1], N_small_r)
                delta_r = r_bins_1[1] - r_bins_1[0] #(r_max - r_bins)/N_small_r
                for j in range(len(self.r)):
                    #start_time = time.time()
                    r_bins_2 = np.linspace(self.r_bins[j], self.r_bins[j+1], N_small_r)
                    
      
                    R1, R2 = np.meshgrid(r_bins_1, r_bins_2)
                    #print(time.time() - start_time)
                    # use b1 ^2 * Pklin in the covariance matrix
                    j0_return = 2./(self.effective_volume * self.number_density * np.pi**2.) *\
                                j0j0.rotation_method_bessel_j0j0(self.ks, self.b1** 2 * self.pk_lin_z0 * self.growth **2., R1, R2)
                    # this one may need to be corrected to include P2, P4, if we are fitting xi0 (or are we fitting xi(r))?
                    j0_return_pk_sq = 1./(self.effective_volume * np.pi**2.) * \
                                        j0j0.rotation_method_bessel_j0j0(self.ks, (self.b1** 2 * self.pk_lin_z0 * self.growth **2.)**2., R1, R2)
                    #print('j0:', time.time() - start_time)
                    
                    #print('new: ', np.diag(j0_return))
                    if i != j:
                        self.covariance_matrix[i][j] = (4 * math.pi)**2. / ((4/3 * math.pi)**2. * \
                                                                        (self.r_bins[i+1]**3 - self.r_bins[i]**3) * \
                                                                            (self.r_bins[j+1]**3 - self.r_bins[j]**3)) \
                                                        * np.sum(R1 **2 * R2 **2 *(j0_return + j0_return_pk_sq) * delta_r * delta_r)
                    #print(self.covariance_matrix[i][j])
                    #print((4 * math.pi)**2. / ((4/3 * math.pi)**2. * (self.r_bins[i+1]**3 - self.r_bins[i]**3) * (self.r_bins[j+1]**3 - self.r_bins[j]**3)) \
                         # * np.sum(R1**2 * R2 **2 * (j0_return) * delta_r * delta_r))
                    
                    if i == j:
                        xi_IRrs_alpha = np.sum(1 / (2 * math.pi**2) * self.ks[:,np.newaxis]**2 * \
                                               special.spherical_jn(0, self.alpha * self.ks[:,np.newaxis] * r_bins_1) \
                                            * np.exp(-self.ks[:,np.newaxis]**2) * self.b1**2 * self.P_IRres[:, np.newaxis] * \
                                            np.gradient(self.ks)[:,np.newaxis], axis=0)
                        #print(time.time() - start_time)
                        dirac_cf_diag = xi_IRrs_alpha / self.delta_r
                        dirac_cf_matrix = np.diag(dirac_cf_diag)
                        dirac_cf_matrix *= 1 / (self.effective_volume * self.number_density**2)
                        dirac_cf_matrix *= 2/(4 * np.pi * r_bins_1**2.)
      
                        a = np.zeros((len(r_bins_1), len(r_bins_2)))
                        np.fill_diagonal(a, 1)
                        dirac_matrix = a
                        dirac_matrix *= 1 / (self.effective_volume * self.number_density**2)
                        dirac_matrix /= self.delta_r
                        dirac_matrix *= 2 / (4 * np.pi * r_bins_1**2.)
                       # print((4 * math.pi)**2. / ((4/3 * math.pi)**2. * (self.r_bins[i+1]**3 - self.r_bins[i]**3) * (self.r_bins[j+1]**3 - self.r_bins[j]**3)) \
                         # * np.sum(R1**2 * R2 **2 * (j0_return) * delta_r * delta_r))
                            
                        #print(np.sum(R1**2 * R2**2 * (j0_return) * delta_r * delta_r))
                        #print(np.sum(R1**2 * R2**2 * (j0_return) * delta_r * delta_r))
                        
                        self.covariance_matrix[i][j] = (4 * math.pi)**2. / ((4/3 * math.pi)**2. * \
                                                                            (self.r_bins[i+1]**3 - self.r_bins[i]**3) * \
                                                                            (self.r_bins[j+1]**3 - self.r_bins[j]**3)) \
                                                        * np.sum(R1 **2 * R2 **2 * (j0_return + j0_return_pk_sq + \
                                                                                    dirac_matrix + dirac_cf_matrix) * \
                                                                 delta_r * delta_r)
                        #print(self.covariance_matrix[i][j])
                        #print((self.r_bins[i+1]**3 - self.r_bins[i]**3) * (self.r_bins[j+1]**3 - self.r_bins[j]**3))
                        #print((self.r_bins[i+1]**3 - self.r_bins[i]**3)**2.)
                        
    # =============================================================================
    #                     if i == 0:
    #                         print(self.covariance_matrix[i][j])
    #                     
    #                    #covariance_in_integrand = j0_return + j0_return_pk_sq + dirac_cf_matrix + dirac_matrix
    #                     #print(np.diag(dirac_matrix))
    #                     #if (i == 0) and (j == 0):
    #                     xi_IRrs_alpha = np.zeros(len(self.r))
    #                     for i in range(len(self.r)):
    #                         self.xi_gg[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * \
    #                                                special.spherical_jn(0, self.alpha * self.ks * self.r_bins[i]) * \
    #                                                    np.exp(-self.ks**2) * self.P_gg * np.gradient(self.ks))
    #                         
    #                     for i in range(len(self.r)):
    #                         xi_IRrs_alpha[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * \
    #                                                   special.spherical_jn(0, self.alpha * self.ks * self.r_bins[i]) * \
    #                                                       np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
    #                     # use b1 ^2 * Pklin in the covariance matrix
    #                     j0_return = 2./(self.effective_volume * self.number_density * np.pi**2.) * \
    #                         j0j0.rotation_method_bessel_j0j0(self.ks, self.b1 ** 2 * self.pk_lin_z0 * \
    #                                                          self.growth **2., self.R1, self.R2)
    #                     # this one may need to be corrected to include P2, P4, if we are fitting xi0 (or are we fitting xi(r))?
    #                     j0_return_pk_sq = 1./(self.effective_volume * np.pi**2.) * \
    #                         j0j0.rotation_method_bessel_j0j0(self.ks, (self.b1 ** 2 * self.pk_lin_z0 * self.growth **2.)**2., \
    #                                                          self.R1, self.R2)
    #                     
    #                     dirac_cf_diag = xi_IRrs_alpha / self.delta_r
    #                     dirac_cf_matrix = np.diag(dirac_cf_diag)
    #                     dirac_cf_matrix *= 1 / (self.effective_volume * self.number_density**2)
    #                     dirac_cf_matrix *= 2/(4 * np.pi * self.r**2.)
    #                     
    #                     a = np.zeros((30,30))
    #                     np.fill_diagonal(a, 1)
    #                     dirac_matrix = a
    #                     dirac_matrix *= 1 / (self.effective_volume * self.number_density**2)
    #                     dirac_matrix /= self.delta_r
    #                     dirac_matrix *= 2 / (4 * np.pi * self.r**2.)
    #                     
    #                     #self.covariance_matrix_old = j0_return + j0_return_pk_sq + dirac_cf_matrix + dirac_matrix
    #                     #print(dirac_matrix[0][0])
    # =============================================================================
                        
                        
                        #self.covariance_matrix[i][j] += dirac_matrix[i][j] + dirac_cf_matrix[i][j]
                    print(self.covariance_matrix[i][j])
                    
            np.savetxt('covariance_matrix_11-26-21.txt', self.covariance_matrix)
            
        
        
    def calc_CF(self):
        self.xi_gg_data = np.zeros(len(self.r))
        print('alpha: ', self.alpha)
        for i in range(len(self.r)):
            self.xi_gg_data[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * \
                                        special.spherical_jn(0, self.alpha * self.ks * self.r[i]) * \
                                            np.exp(-self.ks**2) * self.P_gg * np.gradient(self.ks)) #+ self.covariance_matrix[i]
    
    def get_data(self):
        return self.xi_gg_data
    
    
    def get_covariance_matrix(self):
        return self.covariance_matrix
    
    
    def templates(self):
        self.xi_IRrs = np.zeros(len(self.r))
    
        for i in range(len(self.r)):
            self.xi_IRrs[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * \
            special.spherical_jn(0, self.ks * self.r[i]) * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
    
        return self.xi_IRrs
    
    
    def templates_deriv(self):
        self.xi_IRrs_prime = np.zeros(len(self.r))
    
        for i in range(len(self.r)):
            self.xi_IRrs_prime[i] = -self.r[i] * np.sum(1 / (2 * math.pi**2) * self.ks**3 * special.spherical_jn(1, self.ks * self.r[i]) * np.exp(-self.ks**2)\
                                                         * self.P_IRres * np.gradient(self.ks))
        return self.xi_IRrs_prime
    
    
    def templates_deriv2(self):
        self.xi_IRrs_prime2 = np.zeros(len(self.r))
    
        for i in range(len(self.r)):
            self.xi_IRrs_prime2[i] = self.r[i]**2 * np.sum(1 / (2 * math.pi**2) * self.ks**4 * (special.spherical_jn(2, self.ks * self.r[i]) - (1 / (self.ks * self.r[i])) \
                                                                 * special.spherical_jn(1, self.ks * self.r[i])) * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))    
        return self.xi_IRrs_prime2
    
    
    def templates_deriv3(self):
        self.xi_IRrs_prime3 = np.zeros(len(self.r))
        
        for i in range(len(self.r)):
            self.xi_IRrs_prime3[i] = self.r[i]**3 * np.sum(1. / (2. * math.pi**2) * self.ks**5 * (-1. * \
                                                                special.spherical_jn(3, self.ks * self.r[i]) \
                                                                + (2. / (self.ks * self.r[i])) \
                                                                * special.spherical_jn(2, self.ks * self.r[i])) \
                                                                * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
                
            self.xi_IRrs_prime3[i] += self.r[i] * np.sum(1. / (2. * math.pi**2) * self.ks**3 * ( \
                                                                special.spherical_jn(1, self.ks * self.r[i])) \
                                                                * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
                
            self.xi_IRrs_prime3[i] += self.r[i]**2 * np.sum(1. / (2. * math.pi**2) * self.ks**4 * ( \
                                                                special.spherical_jn(2, self.ks * self.r[i]) \
                                                                + (1. / (self.ks * self.r[i])) \
                                                                * special.spherical_jn(1, self.ks * self.r[i])) \
                                                                * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
        
        return self.xi_IRrs_prime3
    
    
    def get_biases(self):
        return (self.b1 * self.b1)