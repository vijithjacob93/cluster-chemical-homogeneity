import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# class to read, clean, and initialize the data files
class fits_file(object):
    # from __future__ import division
    def __init__(self,filename,filedir):
        import numpy as np
        from astropy.table import Table
        if filedir == 'default':
            filedir = 'C:/Users/vijit/Documents/Repositories/clusterchemistry/clusterchemistry/data/'
        else:
            pass
        self.data = Table.read(filedir+filename, format='fits')
    
    def allCal_init(self,dwarfs_or_giants):
        #remove stars that have -9999 as abundance value
        for i in range(26):
            self.data = self.data[self.data['FELEM'][:,0,i]>-9000] 
        #remove stars that have -9999 as stellar parameters
        for j in [0,1,3,6]:
            self.data = self.data[self.data['FPARAM'][:,j]>-9000]
        # select dwarves/red giants based on the property 'CLASS' in the file
        if dwarfs_or_giants == 'dwarfs':
            cond = np.logical_or.reduce((['Fd' in i for i in self.data['CLASS']],
                ['GKd' in i for i in self.data['CLASS']],['Md' in i for i in self.data['CLASS']]))
        elif dwarfs_or_giants == 'giants':
            cond = np.logical_or(['Mg' in i for i in self.data['CLASS']],['GKg' in i for i in self.data['CLASS']])
        else:
            print("Please enter 'dwarfs' or 'giants'")
        self.data = self.data[cond]
        
    def cluster_members_init(self,dwarfs_or_giants):
        # call in the DR 16 file
        dr_16 = fits_file("allStar-r12-l33.fits","default")
        # merge with distance file to get distances for all APOGEE stars
        distance_data = fits_file("cbj_spectroPhotom_allStar_dr16beta_order.fits","default")        
        
        # get distances
        for i in range(len(distance_data.data)):
            distance_data.data[i]['APOGEE_ID'] = distance_data.data[i]['APOGEE_ID'].strip()
        dr_16.data['D2_med'] = distance_data.data['D2_med']        
        
        from astropy import table
        
        # join the cluster members file with the DR 16 file
#         self.data = dr_16_data
        self.data = table.join(dr_16.data, self.data,keys=['APOGEE_ID'])#['APOGEE_ID','Cluster','no_sigmas_RV','no_sigmas_PM',\
                                                     #'dist_center']
        #define a new column for the SNR bin number and Teff/M_H bin number
        self.data['SNR_bin']=np.nan
        self.data['TEFF_MH_bin']=np.nan
        #remove cluster stars with Teff and M_H == -9999 
        self.data=self.data[np.logical_and(self.data['TEFF']>-9000,self.data['M_H']>-9000)]
        
        # select dwarves/red giants based on the property 'CLASS' in the file
        if dwarfs_or_giants == 'dwarfs':
            self.data=self.data[self.data['M_H']>-1.5]        
            cond=np.logical_or.reduce((['Fd' in i for i in self.data['ASPCAP_CLASS']],\
                ['GKd' in i for i in self.data['ASPCAP_CLASS']],['Md' in i for i in self.data['ASPCAP_CLASS']]))
        elif dwarfs_or_giants == 'giants':
            cond=np.logical_or(['Mg' in i for i in self.data['ASPCAP_CLASS']],\
                ['GKg' in i for i in self.data['ASPCAP_CLASS']])
        else:
            print("Please enter 'dwarfs' or 'giants'")
        self.data = self.data[cond]

	# 
    def field_stars_init(self,dwarfs_or_giants):
        from astropy.table import unique
        # data cleaning steps (Temperature, RV scatter, surface gravity)
        self.data = unique(self.data,keys='APOGEE_ID')
        self.data = self.data[self.data['TEFF']>3600]   
        self.data = self.data[self.data['VSCATTER']<1]
        self.data = self.data[self.data['LOGG']<3]
        # remove stars with 'bad' quality flags
        self.data = self.data[np.logical_and.reduce((self.data['STARFLAG'] & 2**17 == 0, self.data['STARFLAG'] & 2**2 == 0,\
                                                     self.data['STARFLAG'] & 2**3 == 0, self.data['ASPCAPFLAG'] & 2**19 == 0,\
                                                     self.data['ASPCAPFLAG'] & 2**20 == 0,self.data['ASPCAPFLAG'] & 2**23 == 0))]
        # select stars with good signal to noise ratio
        self.data = self.data[self.data['SNR']>50]
        self.data=self.data[np.logical_and(self.data['TEFF']>-9000,self.data['M_H']>-9000)]
        # initialize the SNR and Teff bins
        self.data['SNR_bin']=np.nan
        self.data['TEFF_MH_bin']=np.nan
        
        # select dwarves/red giants based on the property 'CLASS' in the file
        if dwarfs_or_giants == 'dwarfs':
            cond=np.logical_or.reduce((['Fd' in i for i in self.data['ASPCAP_CLASS']],\
                                       ['GKd' in i for i in self.data['ASPCAP_CLASS']],\
                                       ['Md' in i for i in self.data['ASPCAP_CLASS']]))
        elif dwarfs_or_giants == 'giants':
            cond=np.logical_or(['Mg' in i for i in self.data['ASPCAP_CLASS']],\
                               ['GKg' in i for i in self.data['ASPCAP_CLASS']])
        else:
            print("Please enter 'dwarfs' or 'giants'")
        self.data = self.data[cond]
    
    def uncertainties(self):
        elements=['C_FE','CI_FE','N_FE','O_FE','NA_FE','MG_FE','AL_FE','SI_FE','P_FE','S_FE','K_FE','CA_FE','TI_FE',\
          'TIII_FE','V_FE','CR_FE','MN_FE','FE_H','CO_FE','NI_FE',\
            'CU_FE','GE_FE','RB_FE','CE_FE','ND_FE','YB_FE']

        properties=['TEFF','LOGG','M_H','ALPHA_M']

        # initialize columns for the quantities to calculate, i,e., uncertainties for abundances and stellar properties
        for this_element in elements:
            self.data[this_element+"_uncertainty_derived"]=np.nan
            self.data[this_element+"_uncertainty_failure_rate"]=np.nan

        for this_property in properties:
            self.data[this_property+"_uncertainty_derived"]=np.nan
            self.data[this_property+"_uncertainty_failure_rate"]=np.nan
        
# class for each cluster of stars, containing important properties 
class Cluster():
    def __init__(self,name,data_dir):#="",glon=0.0,glat=0.0,radius=0.0):
        from clusterchemistry.functions import Complete_Clusters_func
        Complete_Clusters = Complete_Clusters_func(data_dir)
        clust_cond=Complete_Clusters['NAME']==name
        self.name=name
        # Galactic coordinates
        self.glon=Complete_Clusters[clust_cond]['LII'].values[0]
        self.glat=Complete_Clusters[clust_cond]['BII'].values[0]
        # cluster properties
        self.radius=Complete_Clusters[clust_cond]['CLUSTER_RADIUS'].values[0]
        self.metallicity=Complete_Clusters[clust_cond]['METALLICITY'].values[0]
        self.log_age=Complete_Clusters[clust_cond]['LOG_AGE'].values[0]
        self.status=Complete_Clusters[clust_cond]['CLUSTER_STATUS'].values[0]
        self.distance=Complete_Clusters[clust_cond]['DISTANCE'].values[0]
        self.log_age=Complete_Clusters[clust_cond]['LOG_AGE'].values[0]
        self.log_age_err=Complete_Clusters[clust_cond]['LOG_AGE_ERROR'].values[0]

	# property converting the cluster coordinates from 'float' to 'SkyCoord coordinates'
    @property
    def center(self):
        gs=SkyCoord(l=self.glon*u.degree, b=self.glat*u.degree
                    ,frame='galactic').fk5#convert to ra and dec
        return gs
