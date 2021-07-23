from astropy.io import fits
import pandas as pd
from clusterchemistry.classes import fits_file,Cluster
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from astropy import stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes import SphericalCircle
from scipy import stats

def Complete_Clusters_func():
    # read in Karchenko clusters catalog
    data_directory = '' 
    with fits.open(data_directory+'Cluster_Catalog_Kharchenko_updated.fits') as data:
        Complete_Clusters = pd.DataFrame(data[1].data)
    Complete_Clusters['CLUSTER_RADIUS']=Complete_Clusters['CLUSTER_RADIUS']*60.0
    Complete_Clusters['NAME']=Complete_Clusters['NAME'].str.strip()
#     # condition to identify OCs from other objects
#     oc_cond=Complete_Clusters.CLUSTER_STATUS.values=='              '
#     # #define cluster size as radius * distance
#     # Complete_Clusters['CLUSTER_SIZE']=Complete_Clusters['CENTRAL_RADIUS']*Complete_Clusters['DISTANCE']*np.pi/180
#     #LOWER LIMIT FOR RADIUS
#     Complete_Clusters.loc[np.logical_and.reduce((\
#         Complete_Clusters['CLUSTER_RADIUS'].values<7,Complete_Clusters['DISTANCE'].values<2000)),'CLUSTER_RADIUS']=7.
#     #UPPER LIMIT FOR RADIUS
#     Complete_Clusters.loc[Complete_Clusters['CLUSTER_RADIUS']>100,'CLUSTER_RADIUS']/=2
    return Complete_Clusters


def get_Gaia_PMs(cluster,obj_file_name,h_lims,color_lim):
	# check for existing saved files
    try:
        merged_table_center=pd.read_csv('Files/'+cluster.name+'_merged_center_file')
        merged_table_annulus=pd.read_csv('Files/'+cluster.name+'_merged_annulus_file')
        merged_table_total=pd.read_csv('Files/'+cluster.name+'_merged_total_file')
        print('Found merged files')
    # if not found, download files
    except:
        # call in all APOGEE object files that are present in the sky patch we're looking at
        from astropy.table import Table
        data_obj=Table()
        for file_name in obj_file_name:
            file_name=file_name.strip()
            try:
                try:
                    hdulist1=fits.open('/uufs/chpc.utah.edu/common/home/sdss06/apogeework/apogee/target/'
                                       'apogeeObject/apogeeObject_'+file_name+'.fits',hdu=1)
                except:
                    hdulist1=fits.open('/uufs/chpc.utah.edu/common/home/sdss06/apogeework/apogee/target/'
                                       'apogee2Object/apogee2Object_'+file_name+'.fits',hdu=1)
                data_stack=hdulist1[1].data
                print('Found file '+file_name+'!')
                data_obj=table.vstack([data_obj,Table(data_stack)],join_type='outer')
            except:
                print('Couldn\'t find file '+file_name+'. But it\'s ok!')

        # color-magnitude cuts
        data_obj=data_obj[np.logical_and(data_obj['H']>h_lims[0],data_obj['H']<h_lims[1])]
        data_obj=data_obj[np.subtract(np.subtract(data_obj['J'],data_obj['K']),1.5*data_obj['AK_TARG'])>color_lim]


        # find center and annulus stars in object file
        center_stars_obj=stars_within_radius(data_obj,1*cluster.radius,cluster.center)[0]
#         center_stars_obj=center_stars_obj[data_obj[center_stars_obj]['PMRA_ERR']<8]

        annulus_stars_obj=np.setdiff1d(stars_within_radius(data_obj,2*cluster.radius,cluster.center),
                                       stars_within_radius(data_obj,1.5*cluster.radius,cluster.center))
#         annulus_stars_obj=annulus_stars_obj[data_obj[annulus_stars_obj]['PMRA_ERR']<8]
        
        # all stars in the sky area (2x radius)
        total_stars_obj=stars_within_radius(data_obj,2*cluster.radius,cluster.center)[0]
        # fixing the APOGEE_ID to resemble the 2MASS ID
        for i in range(len(data_obj)):
            data_obj['APOGEE_ID'][i]=data_obj['APOGEE_ID'][i][2:]
            data_obj['APOGEE_ID'][i]=data_obj['APOGEE_ID'][i].strip()

        data_df_annulus=Table(data_obj[annulus_stars_obj]).to_pandas()
        data_df_center=Table(data_obj[center_stars_obj]).to_pandas()
        data_df_total=Table(data_obj[total_stars_obj]).to_pandas()

        # check effect of objectfiles
#         data_df_annulus = data_df_annulus[data_df_annulus['DEC']>40.2]
#         data_df_annulus = data_df_annulus[data_df_annulus['RA']<295.3]
        
        # Gaia TAP query to get all the Gaia stars in the sky area
        try:
            gaia_pm=pd.read_csv('Files/'+cluster.name+'_Gaia_TAP_file')
            print('Found Gaia file')
        except:
            from astroquery.gaia import Gaia
            job = Gaia.launch_job_async("select gaia.ra, gaia.dec, gaia.pmra, gaia.pmra_error, gaia.pmdec, \
             gaia.pmdec_error, gaia.radial_velocity, gaia.radial_velocity_error, gaia.source_id, \
             gaia.PHOT_G_MEAN_MAG,tmass.tmass_oid, tmass.source_id, original_ext_source_id \
            from gaiadr2.gaia_source as gaia \
            inner join gaiadr2.tmass_best_neighbour as tmass \
             on gaia.source_id = tmass.source_id \
            where 1=contains( \
             point('ICRS', gaia.ra, gaia.dec), \
             circle('ICRS', "+str(cluster.center.ra.degree)+", "+str(cluster.center.dec.degree)+", "\
                                        +str(2*cluster.radius/60)+"))")
            gaia_pm=job.get_results()
            gaia_pm['original_ext_source_id']=np.array(gaia_pm['original_ext_source_id']).astype('str')
            gaia_pm=gaia_pm.to_pandas()
#             gaia_pm.to_csv('/uufs/astro.utah.edu/common/home/u1063369/Documents/Membership Pipeline Run 8/Files/allStar-r12-l33/'+\
#                            cluster.name+'_Gaia_TAP_file')

        # join APOGEE object file with Gaia TAP file
        merged_table_annulus=data_df_annulus.merge(gaia_pm,how='inner',left_on='APOGEE_ID',\
                                                   right_on='original_ext_source_id')

        merged_table_center=gaia_pm.merge(data_df_center,how='inner',left_on='original_ext_source_id',\
                                          right_on='APOGEE_ID')
        
        merged_table_total=gaia_pm.merge(data_df_total,how='inner',left_on='original_ext_source_id',\
                                          right_on='APOGEE_ID')

#         merged_table_center.to_csv('/uufs/astro.utah.edu/common/home/u1063369/Documents/'+\
#                 'Membership Pipeline Run 8/Files/allStar-r12-l33/'+cluster.name+'_merged_center_file')
#         merged_table_annulus.to_csv('/uufs/astro.utah.edu/common/home/u1063369/Documents/'+\
#                 'Membership Pipeline Run 8/Files/allStar-r12-l33/'+cluster.name+'_merged_annulus_file')
#         merged_table_total.to_csv('/uufs/astro.utah.edu/common/home/u1063369/Documents/'+\
#                 'Membership Pipeline Run 8/Files/allStar-r12-l33/'+cluster.name+'_merged_total_file')
    return [merged_table_center,merged_table_annulus,merged_table_total]

def stars_within_radius(data_new,cluster_radius,cluster_center):
	# return stars within a given cluster radius
    return np.where(SkyCoord.separation(SkyCoord(data_new['RA']*u.deg,data_new['DEC']*u.deg,frame='icrs'),
                                        cluster_center)<cluster_radius*u.arcmin)

def KDE(x,y,total_stars):
	# kernel density estimation for a property (RV) of stars
    # remove nans
    x=x[np.where(np.logical_and(~np.isnan(x),x<5000))[0]]
    # the standard bandwidth
    if 1.06*np.std(x)*(len(x))**(-1/5)<0.2:
        h=0.2
    elif 4.<1.06*np.std(x)*(len(x))**(-1/5):
        h=4.
    else:
        h=1.06*np.std(x)*(len(x))**(-1/5)
        
    x=x.reshape([len(x),1])
    kde = KernelDensity ( kernel='gaussian' , bandwidth = h)
    kde . fit ( x ) # fit the model to the data
    return np.exp(kde.score_samples(y))# final KDE values

def KDE_2D(data_x,data_y,total_stars):
	# 2D kernel density estimation for a property (PM) of stars
    # avoid TypeError: 'NoneType' object is not iterable
    data_x=data_x[np.where(~np.isnan(data_x))[0]]
    data_y=data_y[np.where(~np.isnan(data_y))[0]]
    data = np.vstack([data_x, data_y]).T
    
    # the standard bandwidth
    h=0.5# 1.06*np.std(data_x)*(len(data_x))**(-1/5)
    kde = KernelDensity ( kernel='gaussian' , bandwidth = h)
    kde . fit ( data ) # fit the model to the data
    x=np.linspace(-20,20,100)
    y=np.linspace(-20,20,100)
    X,Y = np.meshgrid(x,y)
    sample = np.vstack([X.ravel(), Y.ravel()]).T

    return np.exp(kde.score_samples(sample)).reshape(100,100)# the final KDE values

def generate_2D_pdf(parameters):
    # generate 2D PM space for plotting
    x=np.linspace(-20,20,100)
    y=np.linspace(-20,20,100)
    z=np.empty([len(x),len(y)])
    # assign the parameters of the 2D Gaussian
    xo, yo, sigma_x, sigma_y, amplitude, theta, scale=parameters
    for i in range(len(x)):
        for j in range(len(y)):
            xo = float(xo)
            yo = float(yo)    
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            g = amplitude*np.exp( - (a*((x[i]-xo)**2) + 2*b*(x[i]-xo)*(y[j]-yo) + c*((y[j]-yo)**2)))
            z[j,i]=g
    return z
    
def Gaussian_fit(x,data,cluster_name):
	# function fitting a Gaussian distribution to data points using regression
    x=x.reshape(len(x))
    # define Gaussian function
    def gauss(x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    peak_pos=x[np.argmax(data)]
    pars, cov = curve_fit(gauss,x,data,check_finite=False,p0=[peak_pos,2,1],bounds=([-np.inf,0.2,0],
                            [np.inf,6,100]))# fitting with bounds and initial guess

    print('RV center',pars[0]) # print RV center
    
    return pars,cov,x

def Gaussian_fit_2D(line,data,background,cluster_name):
    # function fitting a 2D Gaussian with a scaled offset to data points using regression
    # to fit the signal
    def twoD_Gaussian(Z, xo, yo, sigma_x, sigma_y, amplitude, theta, scale):
        x=Z[0]
        y=Z[1]
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = scale*background +amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()
    
    # to fit the background
    def twoD_Gaussian_background(Z, xo, yo, sigma_x, sigma_y, amplitude, theta):
        x=Z[0]
        y=Z[1]
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    x=np.linspace(-20,20,100)
    y=np.linspace(-20,20,100)
    
    #intial guess for RA ans DEC
    initial_guess_RA=x[np.where(np.max(data)==data)[1]]
    initial_guess_DEC=y[np.where(np.max(data)==data)[0]]

    x, y = np.meshgrid(x, y)
    initial_guess = (float(initial_guess_RA),float(initial_guess_DEC),1.,1.,1.,0.,1.)
    data=data.ravel()
    # center fit
    popt, pcov = curve_fit(twoD_Gaussian, (x, y), data,p0=initial_guess)#,bounds=([0,-np.inf,-np.inf,0.2,0.2,-np.inf,-np.inf],[100,np.inf,np.inf,10,10,np.inf,np.inf]))
    
    #background fit
    background_center_RA=x[np.argmax(background,axis=1)]
    background_center_DEC=y[np.argmax(background,axis=0)]
    background=background.ravel()
    popt_background, pcov_background = curve_fit(twoD_Gaussian_background, (x, y), background)#,bounds=([0,-np.inf,-np.inf,0.2,0.2,-np.inf,-np.inf],[100,np.inf,np.inf,10,10,np.inf,np.inf]))
    #distance between peaks of center and background fits divided by dispersion of center fit
    dist = np.sqrt(((popt[0]-popt_background[0])**2+(popt[1]-popt_background[1])**2)/(popt[2]**2+popt[3]**2))
    
    print('PM RA center',popt[0])
    print('PM DEC center',popt[1])

    return popt,pcov,dist

def prob_member(v_rad,parameters):
    #calculate probability in RV
    return norm.pdf(v_rad,loc=parameters[0],scale=parameters[1])/norm.pdf(parameters[0],loc=parameters[0],scale=parameters[1])

def prob_member_2D(pm,parameters):
    #calculate probability in PM
    xo, yo, sigma_x, sigma_y, amplitude, theta, scale=parameters
    amplitude=1
    
    x=pm[0]
    y=pm[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g

def n_sigma_function(probs):
    #convert probability to no of sigmas
    return np.sqrt(-2*np.log(probs))

def dist_center_function(coords,cluster_center):
    #calculate distance of star from center of cluster
    ra=coords[0]
    dec=coords[1]
    loc=SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')
    dist=SkyCoord.separation(loc,cluster_center)
    return dist.arcmin

def cluster_membership(cluster_name,data_new):
    print(' ')
    try:
        global recovered_cluster_list
        Complete_Clusters=Complete_Clusters_func()
    
        cluster=Cluster(cluster_name)
    
        #Finding center stars and annulus stars
        center_stars=stars_within_radius(data_new,1*cluster.radius,cluster.center)[0]
        annulus_stars=np.setdiff1d(stars_within_radius(data_new,2*cluster.radius,cluster.center),
                                   stars_within_radius(data_new,1.5*cluster.radius,cluster.center))
        total_stars=stars_within_radius(data_new,2*cluster.radius,cluster.center)[0]
        
        data_center_stars=data_new[center_stars]
        data_annulus_stars=data_new[annulus_stars]
        data_total_stars=data_new[total_stars]
        
        #just a random print statement
        print(cluster_name,' Central stars=',len(center_stars),'; Annulus Stars=',len(annulus_stars),
               ' Total stars=',len(total_stars))

        #at least 3 stars in the central region
        if len(data_center_stars)>5:
            
            #find all different fields spanned by stars in this patch of sky
            uniq=np.unique(data_total_stars['FIELD'],return_counts=True)
            #color limits as seen in center stars in APOGEE
            color_lim=np.min(np.subtract(np.subtract(data_total_stars['J'],data_total_stars['K']),
                                         1.5*data_total_stars['AK_TARG']))
            
            #get Gaia center and annulus stars using Gaia TAP query
            phot_gaia_center, phot_gaia_annulus, phot_gaia_total = get_Gaia_PMs(cluster,uniq[0],\
                            [np.min(data_total_stars['H']), np.max(data_total_stars['H'])], color_lim)
            #extract PMs and RVs for the Gaia stars
            #making sure pmra and pmdec from Gaia do not have high errors
            pm_center_err_cond = [np.logical_and(phot_gaia_center['pmra_error']<2.0,\
                                                 phot_gaia_center['pmdec_error']<2.0)][0]
            pm_annulus_err_cond = [np.logical_and(phot_gaia_annulus['pmra_error']<2.0,\
                                                 phot_gaia_annulus['pmdec_error']<2.0)][0]
            PM_RA_center=np.array(phot_gaia_center['pmra'][pm_center_err_cond])
            PM_DEC_center=np.array(phot_gaia_center['pmdec'][pm_center_err_cond])
            PM_RA_annulus=np.array(phot_gaia_annulus['pmra'][pm_annulus_err_cond])
            PM_DEC_annulus=np.array(phot_gaia_annulus['pmdec'][pm_annulus_err_cond])
          
            
            #Fitting the PMs with 2D Gaussian
            size=0.1
            pm_lims=[-150,150]
            x=np.arange(pm_lims[0],pm_lims[1],size)
            x=x.reshape([len(x),1])
            center_f = KDE_2D(PM_RA_center,PM_DEC_center,center_stars) #KDE for the 2D PMs
            annulus_f = KDE_2D(PM_RA_annulus,PM_DEC_annulus,center_stars)
            parameters_PM,cov_PM,peak_separation_PM=Gaussian_fit_2D(x,center_f,annulus_f,cluster_name) #fitting
            #quality of fit for PM
            peak_separation_PM=np.sqrt(((parameters_PM[0]-np.median(PM_RA_annulus[~np.isnan(PM_RA_annulus)]))**2+\
                (parameters_PM[1]-np.median(PM_DEC_annulus[~np.isnan(PM_DEC_annulus)]))**2)/(parameters_PM[2]**2+\
                parameters_PM[3]**2+st.median_absolute_deviation(PM_RA_annulus[~np.isnan(PM_RA_annulus)])**2\
                +st.median_absolute_deviation(PM_DEC_annulus[~np.isnan(PM_DEC_annulus)])**2))
            
            #same for Radial Velocities
            annulus_dens_V_RAD = KDE(data_annulus_stars['VHELIO_AVG'],x,center_stars)            
            center_dens_V_RAD = KDE(data_center_stars['VHELIO_AVG'],x,center_stars)
            subtracted_dens_V_RAD=center_dens_V_RAD-annulus_dens_V_RAD#subtracting annulus distr from central
            subtracted_dens_V_RAD[subtracted_dens_V_RAD<0]=0
            
            parameters_V_RAD,covariances_V_RAD,x=Gaussian_fit(x,subtracted_dens_V_RAD,cluster_name)#fitting
            
            #quality of fit for Radial Velocity
            peak=np.array([np.argmax(subtracted_dens_V_RAD)])
            residual=np.std(subtracted_dens_V_RAD[np.where(np.logical_or(x<x[peak[0]]-6,x>x[peak[0]]+6))])
            Ampl_residual_RV=subtracted_dens_V_RAD[peak[0]]/residual
            peak_separation_RV=abs(parameters_V_RAD[0]-np.median(data_annulus_stars['VHELIO_AVG']))/\
                np.sqrt(parameters_V_RAD[1]**2+st.median_absolute_deviation(data_annulus_stars['VHELIO_AVG'])**2)
            
            #rescaling fit for RV
            fit=norm.pdf(x,parameters_V_RAD[0],parameters_V_RAD[1])
            fit_factor=subtracted_dens_V_RAD[peak[0]]/np.max(fit)
            fit=fit*fit_factor
            normed_fit=fit/max(fit)

            #Finding Probabilities in each Dimension
            prob_RV=np.zeros(len(data_total_stars))
            prob_PM=np.zeros(len(data_total_stars))
            dist_center=np.zeros(len(data_total_stars))

            for count in range(len(data_total_stars['VHELIO_AVG'])):
                prob_RV[count]=prob_member(data_total_stars['VHELIO_AVG'][count],parameters_V_RAD)#RV probability
                dist_center[count]=dist_center_function(list(data_total_stars['RA','DEC'][count]),
                                                        cluster.center)#distance from center of cluster
                i,j=data_total_stars['GAIA_PMRA'][count],data_total_stars['GAIA_PMDEC'][count]
                prob_PM[count]=prob_member_2D([i,j],parameters_PM)#PM probability in a separate function
            
            #calculate number of sigmas from probability
            n_sigmas_RV=n_sigma_function(prob_RV)
            n_sigmas_PM=n_sigma_function(prob_PM)
            
            #this is the new condition using no of sigmas
            w1=1#peak_separation_RV
            w2=1#peak_separation_PM
            A1=1
            A2=1
            prob_total=np.exp((A1*w1*np.log(prob_RV)+A2*w2*np.log(prob_PM))/(A1*w1+A2*w2))
            n_sigmas_total=n_sigma_function(prob_total)
            
            data_new_cluster=data_total_stars[n_sigmas_total<3]

            print('Number of cluster stars=',len(data_new_cluster))

            #define parameters to save in file
            M_H=np.mean(data_new_cluster["M_H"])#Complete_Clusters[Complete_Clusters['NAME']==cluster_name]["METALLICITY"].values
            M_H_dispersion=np.std(data_new_cluster["M_H"][data_new_cluster['M_H']>-9000])#Complete_Clusters[Complete_Clusters['NAME']==cluster_name]["METALLICITY_ERROR"].values
            num_bkgd_stars = len(data_annulus_stars)
            num_cluster_members = len(data_new_cluster)
            velocity_dispersion = np.std(data_new_cluster['VHELIO_AVG'])
            if (np.logical_and(len(data_new_cluster)>3,Ampl_residual_RV>9.5)):
                validation_flag = 'GOOD'
            else:
                validation_flag = 'BAD'

#             save file with total stars including center stars and annulus stars with their no. of sigmas
#             ascii.write([[cluster.name]*len(data_total_stars),data_total_stars['APOGEE_ID'],
#                          data_total_stars['RA'],data_total_stars['DEC'],data_total_stars['GLON'],
#                         data_total_stars['GLAT'],n_sigmas_RV,n_sigmas_PM,n_sigmas_total,dist_center,
#                          [cluster.log_age]*len(data_total_stars)],
#             '/uufs/astro.utah.edu/common/home/u1063369/Documents/Membership Pipeline Run 8/Files/'+cluster_name+
#                         '_members_and_background_Kharchenko.dat', 
#                         names=['Cluster','APOGEE_ID','RA','DEC','GLON','GLAT','no_sigmas_RV','no_sigmas_PM', 
#                                'no_sigmas_total','dist_center','log_age'])
#             save a separate file with cluster parameters
#             ascii.write([[cluster.name],[cluster.status],[cluster.center.ra.degree],\
#                          [cluster.center.dec.degree],[cluster.radius],[cluster.distance],[APOGEE_distance],\
#                          [APOGEE_distance_err],[velocity_dispersion],[cluster.log_age],\
#                          [cluster.log_age_err]\
#                          ,[M_H],[M_H_dispersion],[parameters_V_RAD[0]],[parameters_V_RAD[1]],\
#                          [parameters_V_RAD[2]],[parameters_PM[0]],[parameters_PM[1]],[parameters_PM[2]],\
#                          [parameters_PM[3]],[parameters_PM[4]],[parameters_PM[5]],[parameters_PM[6]],\
#                          [uniq[0][np.argmax(uniq[1])]],[Ampl_residual_RV],[peak_separation_RV],\
#                          [peak_separation_PM],[num_bkgd_stars],[num_cluster_members],[validation_flag]],\
#             '/uufs/astro.utah.edu/common/home/u1063369/Documents/Membership Pipeline Run 8/Files/'+cluster_name+
#                         '_RV_and_PM_fit_parameters_Kharchenko.dat',
#                 names=list(['Cluster_Name','Cluster_Status','Cluster_Center_RA','Cluster_Center_DEC',
#                 'Cluster_Radius','Heliocentric_Distance_Catalog','Heliocentric_Distance_APOGEE',\
#                 'Heliocentric_Distance_err_APOGEE','Velocity_dispersion','log_age','log_age_err','Metallicity',
#                 'Metallicity_dispersion','RV_fit_Mean','RV_fit_Std_Dev','RV_fit_Amplitude','PM_fit_RA Mean',
#                 'PM_fit_DEC_Mean','PM_fit_RA_Std_Dev','PM_fit_DEC_Std_Dev','PM_fit_Amplitude','PM_fit_Theta',
#                 'PM_fit_Background_Scale','Field Used for PMs','Ampl_residual_RV','peak_separation_RV',\
#                 'peak_separation_PM','num_background_stars','num_cluster_members','Validation']))                
        
            n_sigmas_total=n_sigmas_total[n_sigmas_total<3]

            if (np.logical_and(len(data_new_cluster)>3,Ampl_residual_RV>9.5)): 
        
        
            #condition for number of cluster members/RV fit quality
                
                #make directory
                import os
                # define the name of the directory to be created
                path = "/uufs/astro.utah.edu/common/home/u1063369/Documents/Clusters from membership/"\
                        +cluster_name+"/"
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                
                #Plot 'em up!
                bitmask_cond = np.logical_and.reduce((\
                    data_new_cluster['STARFLAG'] & 2**17 == 0, data_new_cluster['STARFLAG'] & 2**2 == 0,\
                    data_new_cluster['STARFLAG'] & 2**3 == 0, data_new_cluster['ASPCAPFLAG'] & 2**19 == 0,\
                    data_new_cluster['ASPCAPFLAG'] & 2**20 == 0, data_new_cluster['ASPCAPFLAG'] & 2**23 == 0))

                data_new_cluster_bitmask = data_new_cluster[bitmask_cond]
                n_sigmas_total_bitmask = n_sigmas_total[bitmask_cond]

                fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7,10))
                pm_range_x = [-10,5]
                pm_range_y = [-10,5]
                inner_color = 'blue'
                annulus_color = 'green'
                field_color = 'gray'
                #calculating no of sigmas from probability

                labelsize=12
                pm_contourf_levels = [0.1,0.25,0.5,0.75,0.9,1.0]                
                pm_contour_levels = [0.5,0.75,0.9,1.0]                
                
                #to flip stars in RA<0
                if (np.logical_or(cluster.center.ra.arcmin-10*cluster.radius<0,\
                    cluster.center.ra.arcmin+10*cluster.radius>360.0*60.0)):
                    phot_gaia_center.loc[phot_gaia_center['ra']>180.0,'ra']=phot_gaia_center[phot_gaia_center\
                                                                                ['ra']>180.0]['ra']-360.0
                    phot_gaia_annulus.loc[phot_gaia_annulus['ra']>180.0,'ra']=phot_gaia_annulus\
                                                                [phot_gaia_annulus['ra']>180.0]['ra']-360.0
                    data_center_stars['RA'][data_center_stars['RA']>180.0]-=360.0
                    data_annulus_stars['RA'][data_annulus_stars['RA']>180.0]-=360.0
                    data_new_cluster['RA'][data_new_cluster['RA']>180.0]-=360.0
                    data_total_stars['RA'][data_total_stars['RA']>180.0]-=360.0
                else:
                    circ_center = SphericalCircle((cluster.center.ra.degree*u.deg, cluster.center.dec.degree*\
                                u.deg),1.0*cluster.radius * u.arcmin, edgecolor=inner_color, facecolor='none')
                    circ_annulus_1 = SphericalCircle((cluster.center.ra.degree*u.deg, cluster.center.dec.degree*\
                                u.deg),1.5*cluster.radius * u.arcmin, edgecolor=annulus_color, facecolor='none')
                    circ_annulus_2 = SphericalCircle((cluster.center.ra.degree*u.deg, cluster.center.dec.degree*\
                                u.deg),2.0*cluster.radius * u.arcmin, edgecolor=annulus_color, facecolor='none')
                    ax[0,0].add_patch(circ_center)
                    ax[0,0].add_patch(circ_annulus_1)
                    ax[0,0].add_patch(circ_annulus_2)
                    circ_center = SphericalCircle((cluster.center.ra.degree*u.deg, cluster.center.dec.degree*\
                                u.deg),1.0*cluster.radius * u.arcmin, edgecolor=inner_color, facecolor='none')
                    circ_annulus_1 = SphericalCircle((cluster.center.ra.degree*u.deg, cluster.center.dec.degree*\
                                u.deg),1.5*cluster.radius * u.arcmin, edgecolor=annulus_color, facecolor='none')
                    circ_annulus_2 = SphericalCircle((cluster.center.ra.degree*u.deg, cluster.center.dec.degree*\
                                u.deg),2.0*cluster.radius * u.arcmin, edgecolor=annulus_color, facecolor='none')
                    ax[0,1].add_patch(circ_center)
                    ax[0,1].add_patch(circ_annulus_1)
                    ax[0,1].add_patch(circ_annulus_2)
                
                
                #Sky Plot
                edgewidth = 0.1
                ax[0,0].scatter(phot_gaia_center['ra'], phot_gaia_center['dec'], s=3, marker='.', edgecolors='none',
                           c=field_color,label='All Phot stars',zorder=-20)
                ax[0,1].scatter(phot_gaia_center['ra'], phot_gaia_center['dec'], s=3, marker='.', edgecolors='none',
                           c=field_color,label='All Phot stars',zorder=-20)
                ax[0,0].scatter(phot_gaia_annulus['ra'], phot_gaia_annulus['dec'], s=3, marker='.', edgecolors='none',
                           c=field_color,label='All Phot stars',zorder=-20)
                ax[0,1].scatter(phot_gaia_annulus['ra'], phot_gaia_annulus['dec'], s=3, marker='.', edgecolors='none',
                           c=field_color,label='All Phot stars',zorder=-20)
                
     
                ax[0,0].scatter(data_total_stars['RA'], data_total_stars['DEC'], s=15, marker='^',alpha = 0.7,
                           edgecolors='k',linewidths = edgewidth,c=field_color,label='Center stars',zorder= 10)

                ax[0,0].scatter(data_center_stars['RA'], data_center_stars['DEC'], s=15, marker='^',alpha = 0.7,
                           edgecolors='k',linewidths = edgewidth,c=inner_color,label='Center stars',zorder= 10)
                
                ax[0,0].scatter(data_annulus_stars['RA'], data_annulus_stars['DEC'], s=15, marker='^',alpha = 0.7,
                           edgecolors='k',linewidths = edgewidth, c=annulus_color,label='Annulus stars',zorder= 10)
                
                ax[0,1].scatter(data_new_cluster['RA'],\
                                data_new_cluster['DEC'],c='m',#=n_sigmas_total,
                                s=30, marker = "^",
                              vmin=0.,vmax=3.,edgecolors='k',linewidths = edgewidth,cmap='spring', \
                                label='APOGEE Cluster Members',zorder=10)

                ax[0,0].set_xlabel('RA [deg]', fontsize=labelsize)
                ax[0,0].set_ylabel('Dec [deg]', fontsize=labelsize)
                ax[0,1].set_xlabel('RA [deg]', fontsize=labelsize)
                ax[0,1].set_ylabel('Dec [deg]', fontsize=labelsize)

                #RV plo
                ax[1,0].plot(x,center_dens_V_RAD,color=inner_color,alpha=0.8,label='Center stars')
                ax[1,0].plot(x,annulus_dens_V_RAD,color=annulus_color,alpha=0.8, label='Annulus stars')
                ax[1,0].plot(x,subtracted_dens_V_RAD,color='k',linewidth=1.2,alpha=0.8, \
                           label='Subtracted distribution')
                ax[1,0].plot(x,fit,label='Fit',color='red')
                ax[1,0].set_xlim([parameters_V_RAD[0]-25,parameters_V_RAD[0]+25])
                ax[1,0].set_ylim([0,max(fit)+0.05])
                ax[1,0].set_xlabel('Radial Velocity [km/s]', fontsize=labelsize)
                ax[1,0].set_ylabel('Fraction of Stars', fontsize=labelsize)
                if (~np.isnan(Complete_Clusters[Complete_Clusters['NAME']==cluster_name]['RAD_VEL'].values[0])):
                    ax[1,0].axvline(x=Complete_Clusters[Complete_Clusters['NAME']==cluster_name]\
                                  ['RAD_VEL'].values[0],linestyle='--')

                #PM plot
                #define limits for PM plots
                xmin, xmax = -20, 20
                ymin, ymax = -20, 20

                ax[1,1].contourf(annulus_f/np.max(annulus_f),
                           extent=[xmin,xmax,ymin,ymax],
                           levels=pm_contourf_levels, cmap='Greens',alpha=0.6)
                ax[1,1].contour(annulus_f/np.max(annulus_f),
                           extent=[xmin,xmax,ymin,ymax],
                           levels=pm_contour_levels, colors='Green',alpha=0.8)
                
                ax[1,1].contourf(center_f/np.max(center_f),
                           extent=[xmin,xmax,ymin,ymax],
                           levels=pm_contourf_levels, cmap='Blues',alpha=0.6)
                ax[1,1].contour(center_f/np.max(center_f),
                           extent=[xmin,xmax,ymin,ymax],
                           levels=pm_contour_levels, colors='Blue',alpha=0.8)
                
                pdf_PM_cluster=generate_2D_pdf(parameters_PM)
                ax[1,1].contour(pdf_PM_cluster/np.max(pdf_PM_cluster),
                           extent=[xmin,xmax,ymin,ymax],
                           levels=pm_contour_levels, colors='Red',alpha=0.6)
                if (np.logical_and(~np.isnan(Complete_Clusters[Complete_Clusters['NAME']==cluster_name]\
                    ['PM_RA'].values[0]),~np.isnan(Complete_Clusters[Complete_Clusters['NAME']==cluster_name]\
                        ['PM_DEC'].values[0]))):
                    ax[1,1].plot(Complete_Clusters[Complete_Clusters['NAME']==cluster_name]['PM_RA'],
                                Complete_Clusters[Complete_Clusters['NAME']==cluster_name]['PM_DEC'],'k+') 
                    

                ax[1,1].set_xlabel('RA Proper Motion[mas/yr]', fontsize=labelsize)
                ax[1,1].set_ylabel('Dec Proper Motion[mas/yr]', fontsize=labelsize)
                ax[1,1].set_xlim([parameters_PM[0]-6,parameters_PM[0]+6])
                ax[1,1].set_ylim([parameters_PM[1]-6,parameters_PM[1]+6])

                #CMD Isochrone from Padova can be added (done here). Convert it to app mag.
                cluster_mean_mh,cluster_std_mh=np.mean(data_new_cluster[data_new_cluster['M_H']>-100]['M_H'])\
                        ,np.std(data_new_cluster[data_new_cluster['M_H']>-100]['M_H'])

                    
                iso_data=pd.read_csv('all_isochrones.dat',skiprows=11,delim_whitespace=True,comment="#")
                iso_data = iso_data[np.isclose(iso_data["MH"], cluster_mean_mh, atol=0.05)]
                iso_data = iso_data[np.isclose(iso_data["logAge"], cluster.log_age, atol=0.125)]

                print('age = ', cluster.log_age)
                h=iso_data['Hmag']+5*np.log10(Complete_Clusters[Complete_Clusters['NAME']==cluster_name]['DISTANCE']\
                                              .values/10)
                
                j_k = np.subtract(iso_data['Jmag'],iso_data['Kmag']) + float(Complete_Clusters\
                    [Complete_Clusters['NAME']==cluster_name]['E_JK'])

                #bitmasks before plotting diasgnostic plots
                ax[2,0].plot(j_k[:-2],h[:-2],c='k',label='Isochrone',\
                           zorder=-15)#,edgecolors='face',cmap = 'Accent')

                ax[2,0].scatter(phot_gaia_total['J']-phot_gaia_total['K'],#-2.11*phot_gaia_total['AK_TARG'], 
                           phot_gaia_total['H'], s=3, marker='.', c=field_color,edgecolors='none',\
                                      label='All Phot stars',zorder=-20)
                
                im=ax[2,0].scatter(data_new_cluster['J'][data_new_cluster['M_H']>-100]-\
                                 data_new_cluster['K'][data_new_cluster['M_H']>-100],\
                                 #-2.11*data_new_cluster['AK_TARG'][data_new_cluster['M_H']>-100],\
                           data_new_cluster['H'][data_new_cluster['M_H']>-100], edgecolors='k',\
                                 c='m',#n_sigmas_total_bitmask[data_new_cluster_bitmask['M_H']>-100],\
                                   s=30, \
                                 marker = '^',linewidths = edgewidth,\
                           vmin=0.0,vmax=3.0,cmap='spring',label='original',zorder = -10, alpha = 1.0)  
                
                ax[2,0].set_xlim([min(phot_gaia_center['J']-phot_gaia_center['K'])-0.5,
                                max(phot_gaia_center['J']-phot_gaia_center['K'])+0.5])
                ax[2,0].set_ylim([14,min(phot_gaia_center['H'])-1])
                ax[2,0].set_xlabel(r'(J-K$_s$)', fontsize=labelsize)
                ax[2,0].set_ylabel('H', fontsize=labelsize)
            
                #no of sigmas in RV and PM
                ax[2,1].hist(data_new_cluster_bitmask['M_H'],bins=np.arange(-1.0,0.5,0.05)\
                           ,color='m',label='Cluster stars',alpha=0.5)
                ax[2,1].hist(data_annulus_stars['M_H'],bins=np.arange(-1.0,0.5,0.05)\
                           ,color=annulus_color,label='Background stars',alpha=0.5)
                ax[2,1].set_xlabel('Metallicity', fontsize=labelsize)
                ax[2,1].set_ylabel('Number of stars', fontsize=labelsize)
                AD_test = stats.anderson_ksamp([data_new_cluster['M_H'],data_annulus_stars['M_H']])
                
                plt.figtext(0.14,0.95,'(a)', fontsize = 13)
                plt.figtext(0.63,0.95,'(b)', fontsize = 13)
                plt.figtext(0.14,0.62,'(c)', fontsize = 13)
                plt.figtext(0.63,0.62,'(d)', fontsize = 13)
                plt.figtext(0.14,0.29,'(e)', fontsize = 13)
                plt.figtext(0.63,0.29,'(f)', fontsize = 13)
                
                fig.suptitle(cluster_name,y = 0.03, fontsize=labelsize, fontweight = 'bold')
                fig.tight_layout()
                
#                 fig.savefig('/uufs/astro.utah.edu/common/home/u1063369/Documents/Membership_Pipeline_Run_8/'
#                         'Plots/Paper/Member_selection_diagnostic plots:'+cluster_name+'.png',dpi=200)
                plt.show()
                                
                return 1.
            else:
                print('Number of cluster stars is less (UNRECOVERED)')
                return 0.
        else:
            print('Number of center stars is less')
            return 0.
#     except Exception:
#         print('Exception')
#         return 0.
    except ValueError:
        print('ValueError')
        return 0.
    except RuntimeError:
        print('RuntimeError')
        return 0.
    except IndexError:
        print('IndexError')
        return 0.
#     except IOError:
#         print('IOError')
#         return 0.
#     except KeyError:
#         print('KeyError')
#         return 0.
#     except TypeError:
#         print('TypeError')
#         return 0.
    except ZeroDivisionError:
        print('ZeroDivisionError')
        return 0.
#     except :
#         print('Unexpected Error')
#         return 0.



#Voronoi binning functions
from time import perf_counter
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance, cKDTree
from scipy import ndimage

#----------------------------------------------------------------------------

def _sn_func(index, signal=None, noise=None):
    """
    Default function to calculate the S/N of a bin with spaxels "index".

    The Voronoi binning algorithm does not require this function to have a
    specific form and this default one can be changed by the user if needed
    by passing a different function as

        ... = voronoi_2d_binning(..., sn_func=sn_func)

    The S/N returned by sn_func() does not need to be an analytic
    function of S and N.

    There is also no need for sn_func() to return the actual S/N.
    Instead sn_func() could return any quantity the user needs to equalize.

    For example sn_func() could be a procedure which uses ppxf to measure
    the velocity dispersion from the coadded spectrum of spaxels "index"
    and returns the relative error in the dispersion.

    Of course an analytic approximation of S/N, like the one below,
    speeds up the calculation.

    :param index: integer vector of length N containing the indices of
        the spaxels for which the combined S/N has to be returned.
        The indices refer to elements of the vectors signal and noise.
    :param signal: vector of length M>N with the signal of all spaxels.
    :param noise: vector of length M>N with the noise of all spaxels.
    :return: scalar S/N or another quantity that needs to be equalized.
    """

    sn = np.sum(signal[index])#/np.sqrt(np.sum(noise[index]**2))

    # The following commented line illustrates, as an example, how one
    # would include the effect of spatial covariance using the empirical
    # Eq.(1) from http://adsabs.harvard.edu/abs/2015A%26A...576A.135G
    # Note however that the formula is not accurate for large bins.
    #
    # sn /= 1 + 1.07*np.log10(index.size)

    
    return  sn

#----------------------------------------------------------------------

def voronoi_tessellation(x, y, xnode, ynode, scale):
    """
    Computes (Weighted) Voronoi Tessellation of the pixels grid

    """
    if scale[0] == 1:  # non-weighted VT
        tree = cKDTree(np.column_stack([xnode, ynode]))
        classe = tree.query(np.column_stack([x, y]))[1]
    else:
        if x.size < 1e4:
            classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2)/scale**2, axis=1)
        else:  # use for loop to reduce memory usage
            classe = np.zeros(x.size, dtype=int)
            for j, (xj, yj) in enumerate(zip(x, y)):
                classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2)/scale**2)

    return classe

#----------------------------------------------------------------------

def _roundness(x, y, pixelSize):
    """
    Implements equation (5) of Cappellari & Copin (2003)

    """
    n = x.size
    equivalentRadius = np.sqrt(n/np.pi)*pixelSize
    xBar, yBar = np.mean(x), np.mean(y)  # Geometric centroid here!
    maxDistance = np.sqrt(np.max((x - xBar)**2 + (y - yBar)**2))
    roundness = maxDistance/equivalentRadius - 1.

    return roundness

#----------------------------------------------------------------------

def _accretion(x, y, signal, noise, targetSN, pixelsize, quiet, sn_func):
    """
    Implements steps (i)-(v) in section 5.1 of Cappellari & Copin (2003)

    """
    n = x.size
    classe = np.zeros(n, dtype=int)  # will contain the bin number of each given pixel
    good = np.zeros(n, dtype=bool)   # will contain 1 if the bin has been accepted as good

    # For each point, find the distance to all other points and select the minimum.
    # This is a robust but slow way of determining the pixel size of unbinned data.
    #
    if pixelsize is None:
        if x.size < 1e4:
            pixelsize = np.min(distance.pdist(np.column_stack([x, y])))
            print(pixelsize)
        else:
            raise ValueError("Dataset is large: Provide `pixelsize`")

    currentBin = np.argmax(signal/noise)  # Start from the pixel with highest S/N
    SN = sn_func(currentBin, signal, noise)

    # Rough estimate of the expected final bins number.
    # This value is only used to give an idea of the expected
    # remaining computation time when binning very big dataset.
    #
    w = signal/noise < targetSN
    maxnum = int(np.sum((signal[w]/noise[w])**2)/targetSN**2 + np.sum(~w))

    # The first bin will be assigned CLASS = 1
    # With N pixels there will be at most N bins
    #
    for ind in range(1, n+1):

        if not quiet:
            print(ind, ' / ', maxnum)

        classe[currentBin] = ind  # Here currentBin is still made of one pixel
        xBar, yBar = x[currentBin], y[currentBin]    # Centroid of one pixels

        while True:

            if np.all(classe):
                break  # Stops if all pixels are binned

            # Find the unbinned pixel closest to the centroid of the current bin
            #
            unBinned = np.flatnonzero(classe == 0)
            k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)

            # (1) Find the distance from the closest pixel to the current bin
            #
            minDist = np.min((x[currentBin] - x[unBinned[k]])**2 + (y[currentBin] - y[unBinned[k]])**2)

            # (2) Estimate the `roundness' of the POSSIBLE new bin
            #
            nextBin = np.append(currentBin, unBinned[k])
            roundness = _roundness(x[nextBin], y[nextBin], pixelsize)

            # (3) Compute the S/N one would obtain by adding
            # the CANDIDATE pixel to the current bin
            #
            SNOld = SN
            SN = sn_func(nextBin, signal, noise)

            # Test whether (1) the CANDIDATE pixel is connected to the
            # current bin, (2) whether the POSSIBLE new bin is round enough
            # and (3) whether the resulting S/N would get closer to targetSN
            #
            if (np.sqrt(minDist) > 1.2*pixelsize or roundness > 0.3
                or abs(SN - targetSN) > abs(SNOld - targetSN) or SNOld > SN):
                if SNOld > 0.8*targetSN:
                    good[currentBin] = 1
                break

            # If all the above 3 tests are negative then accept the CANDIDATE
            # pixel, add it to the current bin, and continue accreting pixels
            #
            classe[unBinned[k]] = ind
            currentBin = nextBin

            # Update the centroid of the current bin
            #
            xBar, yBar = np.mean(x[currentBin]), np.mean(y[currentBin])

        # Get the centroid of all the binned pixels
        #
        binned = classe > 0
        if np.all(binned):
            break  # Stop if all pixels are binned
        xBar, yBar = np.mean(x[binned]), np.mean(y[binned])

        # Find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel.
        #
        unBinned = np.flatnonzero(classe == 0)
        if sn_func(unBinned, signal, noise) < targetSN:
            break  # Stops if the remaining pixels do not have enough capacity
        k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)
        currentBin = unBinned[k]    # The bin is initially made of one pixel
        SN = sn_func(currentBin, signal, noise)

    classe *= good  # Set to zero all bins that did not reach the target S/N

    return classe, pixelsize

#----------------------------------------------------------------------------

def _reassign_bad_bins(classe, x, y):
    """
    Implements steps (vi)-(vii) in section 5.1 of Cappellari & Copin (2003)

    """
    # Find the centroid of all successful bins.
    # CLASS = 0 are unbinned pixels which are excluded.
    #
    good = np.unique(classe[classe > 0])
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    # Reassign pixels of bins with S/N < targetSN
    # to the closest centroid of a good bin
    #
    bad = classe == 0
    index = voronoi_tessellation(x[bad], y[bad], xnode, ynode, [1])
    classe[bad] = good[index]

    # Recompute all centroids of the reassigned bins.
    # These will be used as starting points for the CVT.
    #
    good = np.unique(classe)
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    return xnode, ynode

#----------------------------------------------------------------------------

def _cvt_equal_mass(x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt):
    """
    Implements the modified Lloyd algorithm
    in section 4.1 of Cappellari & Copin (2003).

    NB: When the keyword WVT is set this routine includes
    the modification proposed by Diehl & Statler (2006).

    """
    dens2 = (signal/noise)**4     # See beginning of section 4.1 of CC03
    scale = np.ones_like(xnode)   # Start with the same scale length for all bins

    for it in range(1, xnode.size):  # Do at most xnode.size iterations

        xnode_old, ynode_old = xnode.copy(), ynode.copy()
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)

        # Computes centroids of the bins, weighted by dens**2.
        # Exponent 2 on the density produces equal-mass Voronoi bins.
        # The geometric centroids are computed if WVT keyword is set.
        #
        good = np.unique(classe)
        if wvt:
            for k in good:
                index = np.flatnonzero(classe == k)   # Find subscripts of pixels in bin k.
                xnode[k], ynode[k] = np.mean(x[index]), np.mean(y[index])
                sn = sn_func(index, signal, noise)
                scale[k] = np.sqrt(index.size/sn)  # Eq. (4) of Diehl & Statler (2006)
        else:
            mass = ndimage.sum(dens2, labels=classe, index=good)
            xnode = ndimage.sum(x*dens2, labels=classe, index=good)/mass
            ynode = ndimage.sum(y*dens2, labels=classe, index=good)/mass

        diff2 = np.sum((xnode - xnode_old)**2 + (ynode - ynode_old)**2)
        diff = np.sqrt(diff2)/pixelsize

        if not quiet:
            print('Iter: %4i  Diff: %.4g' % (it, diff))

        if diff < 0.1:
            break

    # If coordinates have changed, re-compute (Weighted) Voronoi Tessellation of the pixels grid
    #
    if diff > 0:
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)
        good = np.unique(classe)  # Check for zero-size Voronoi bins

    # Only return the generators and scales of the nonzero Voronoi bins

    return xnode[good], ynode[good], scale[good], it

#-----------------------------------------------------------------------

def _compute_useful_bin_quantities(x, y, signal, noise, xnode, ynode, scale, sn_func):
    """
    Recomputes (Weighted) Voronoi Tessellation of the pixels grid to make sure
    that the class number corresponds to the proper Voronoi generator.
    This is done to take into account possible zero-size Voronoi bins
    in output from the previous CVT (or WVT).

    """
    # classe will contain the bin number of each given pixel
    classe = voronoi_tessellation(x, y, xnode, ynode, scale)

    # At the end of the computation evaluate the bin luminosity-weighted
    # centroids (xbar, ybar) and the corresponding final S/N of each bin.
    #
    good = np.unique(classe)
    xbar = ndimage.mean(x, labels=classe, index=good)
    ybar = ndimage.mean(y, labels=classe, index=good)
    area = np.bincount(classe)
    sn = np.empty_like(xnode)
    for k in good:
        index = np.flatnonzero(classe == k)   # index of pixels in bin k.
        sn[k] = sn_func(index, signal, noise)

    return classe, xbar, ybar, sn, area

#-----------------------------------------------------------------------

def _display_pixels(x, y, counts, pixelsize):
    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.

    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
    img[j, k] = counts

    plt.imshow(np.rot90(img), interpolation='nearest', cmap='prism',
               extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                       ymin - pixelsize/2, ymax + pixelsize/2],aspect='auto')

#----------------------------------------------------------------------

def voronoi_2d_binning(x, y, signal, noise, targetSN, cvt=True,
                         pixelsize=None, plot=True, quiet=True,
                         sn_func=None, wvt=True):
    """
    PURPOSE:
          Perform adaptive spatial binning of Integral-Field Spectroscopic
          (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
          This method is required for the proper analysis of IFS
          observations, but can also be used for standard photometric
          imagery or any other two-dimensional data.
          This program precisely implements the algorithm described in
          section 5.1 of the reference below.

    EXPLANATION:
          Further information on VORONOI_2D_BINNING algorithm can be found in
          Cappellari M., Copin Y., 2003, MNRAS, 342, 345

    CALLING SEQUENCE:

        binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetSN,
                               cvt=True, pixelsize=None, plot=True,
                               quiet=True, sn_func=None, wvt=True)

    """
    # This is the main program that has to be called from external programs.
    # It simply calls in sequence the different steps of the algorithms
    # and optionally plots the results at the end of the calculation.

    assert x.size == y.size == signal.size == noise.size, \
        'Input vectors (x, y, signal, noise) must have the same size'
    assert np.all((noise > 0) & np.isfinite(noise)), \
        'NOISE must be positive and finite'

    if sn_func is None:
        sn_func = _sn_func

    # Perform basic tests to catch common input errors
    #
    if sn_func(np.flatnonzero(noise > 0), signal, noise) < targetSN:
        raise ValueError("""Not enough S/N in the whole set of pixels.
            Many pixels may have noise but virtually no signal.
            They should not be included in the set to bin,
            or the pixels should be optimally weighted.
            See Cappellari & Copin (2003, Sec.2.1) and README file.""")
    if np.min(signal/noise) > targetSN:
        raise ValueError('All pixels have enough S/N and binning is not needed')

    t1 = perf_counter()
    if not quiet:
        print('Bin-accretion...')
    classe, pixelsize = _accretion(
        x, y, signal, noise, targetSN, pixelsize, quiet, sn_func)
    if not quiet:
        print(np.max(classe), ' initial bins.')
        print('Reassign bad bins...')
    xnode, ynode = _reassign_bad_bins(classe, x, y)
    if not quiet:
        print(xnode.size, ' good bins.')
    t2 = perf_counter()
    if cvt:
        if not quiet:
            print('Modified Lloyd algorithm...')
        xnode, ynode, scale, it = _cvt_equal_mass(
            x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt)
        if not quiet:
            print(it - 1, ' iterations.')
    else:
        scale = np.ones_like(xnode)
    classe, xBar, yBar, sn, area = _compute_useful_bin_quantities(
        x, y, signal, noise, xnode, ynode, scale, sn_func)
    single = area == 1
    t3 = perf_counter()
    if not quiet:
        print('Unbinned pixels: ', np.sum(single), ' / ', x.size)
        print('Fractional S/N scatter (%):', np.std(sn[~single] - targetSN, ddof=1)/targetSN*100)
        print('Elapsed time accretion: %.2f seconds' % (t2 - t1))
        print('Elapsed time optimization: %.2f seconds' % (t3 - t2))

#     if plot:
#         plt.clf()
#         plt.subplot(211)
#         rnd = np.argsort(np.random.random(xnode.size))  # Randomize bin colors
#         _display_pixels(x, y, rnd[classe], pixelsize)
#         plt.plot(xnode, ynode, '+w', scalex=False, scaley=False) # do not rescale after imshow()
#         plt.ylabel('Whitened Teff (K)')
#         plt.xlabel('Whitened [M/H]')
#         plt.title('Map of Voronoi bins (dwarves)')

#         plt.subplot(212)
#         rad = np.sqrt(xBar**2 + yBar**2)  # Use centroids, NOT generators
#         plt.plot(np.sqrt(x**2 + y**2), signal/noise, ',k')
#         if np.any(single):
#             plt.plot(rad[single], sn[single], 'xb', label='Not binned')
#         plt.plot(rad[~single], sn[~single], 'or', label='Voronoi bins')
#         plt.xlabel('Whitened Teff (K)')
#         plt.ylabel('Number of duplicates (red_giants)')
#         plt.axis([np.min(rad), np.max(rad), 0, np.max(sn)*1.05])  # x0, x1, y0, y1
#         plt.axhline(targetSN)
#         plt.legend()
#         plt.savefig('Voronoi_binning_duplicates_red_giants.png',dpi=300)

    return classe, xnode, ynode, xBar, yBar, sn, area, scale



#!/usr/bin/env python

"""
Copyright (C) 2003-2014, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

    V1.0.0: Michele Cappellari, Vicenza, 13 February 2003
    V1.0.1: Use astro library routines to read and write files.
        MC, Leiden, 24 July 2003
    V2.0.0: Translated from IDL into Python. MC, London, 19 March 2014
    V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
    V2.0.2: Make files paths relative to this file, to run the example from
        any directory. MC, Oxford, 23 January 2017
    V2.0.3: Changed imports for vorbin as package. 
        Make file paths relative to the vorbin package to be able to run the
        example unchanged from any directory. MC, Oxford, 17 April 2018    
    V2.0.4: Dropped legacy Python 2.7 support. MC, Oxford, 10 May 2018

"""

from os import path
import numpy as np
# import matplotlib.pyplot as plt

import vorbin
# from vorbin.voronoi_2d_binning import voronoi_2d_binning

#-----------------------------------------------------------------------------

def voronoi_binning_example(df_func, targetSN):
    """
    Usage example for the procedure VORONOI_2D_BINNING.

    It is assumed below that the file voronoi_2d_binning_example.txt
    resides in the current directory. Here columns 1-4 of the text file
    contain respectively the x, y coordinates of each SAURON lens
    and the corresponding Signal and Noise.

    """
    file_dir = path.dirname(path.realpath(vorbin.__file__))  # path of vorbin
    no_pairs=np.array([1.0]*len(df_func))
    noise_no_pairs=np.array([1.0]*len(df_func))
    #input the metallicity and temperature after whitening the data
    x, y, signal, noise = (df_func['MH'].values-np.mean(df_func['MH'].values))/np.std(df_func['MH'].values)\
                          ,(df_func['TEFF'].values-np.mean(df_func['TEFF'].values))/np.std(df_func['TEFF'].values)\
                           ,no_pairs,noise_no_pairs#np.loadtxt(file_dir + '/voronoi_2d_binning_example_input.txt').T
#     targetSN = 50.0

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=0,pixelsize=0.2)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
#     np.savetxt('Voronoi_binning_duplicates_red_giants.txt', np.column_stack([x, y, binNum]),
#                fmt=b'%10.6f %10.6f %8i')
#     np.savetxt('Voronoi_bin_centers_and_pop_duplicates_red_giants.txt', np.column_stack([xNode, yNode, sn]),
#                fmt=b'%10.6f %10.6f %8i')
    
    return binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale
#-----------------------------------------------------------------------------


