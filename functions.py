from astropy.io import fits
import pandas as pd
from classes import fits_file,Cluster
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def Complete_Clusters():
    # read in Karchenko clusters catalog
    data_directory = 'Data/' 
    with fits.open(data_directory+'Cluster_Catalog_Kharchenko_updated.fits') as data:
        Complete_Clusters = pd.DataFrame(data[1].data)
    Complete_Clusters['CLUSTER_RADIUS']=Complete_Clusters['CLUSTER_RADIUS']*60.0
    Complete_Clusters['NAME']=Complete_Clusters['NAME'].str.strip()
    return Complete_Clusters


def get_Gaia_PMs(cluster,obj_file_name,h_lims,color_lim):
	# check for existing saved files
    data_directory = 'Data/'	
    try:
        merged_table_center=pd.read_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_merged_center_file')
        merged_table_annulus=pd.read_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_merged_annulus_file')
        merged_table_total=pd.read_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_merged_total_file')
        print('Found merged files')
    # if not found, download files
    except:
        # call in all APOGEE object files that are present in the sky patch we're looking at
        from astropy.table import Table
        data_obj=Table()
        for file_name in obj_file_name:
            file_name=file_name.strip()
            try:
                hdulist1=fits.open(data_directory+'apogeeObject_'+file_name+'.fits',hdu=1)                    
                data_stack=hdulist1[1].data
                print('Found file '+file_name+'!')
                data_obj=table.vstack([data_obj,Table(data_stack)],join_type='outer')
            except:
                print('Couldn\'t find file '+file_name+'. But that\'s ok!')

        # color-magnitude cuts
        data_obj=data_obj[np.logical_and(data_obj['H']>h_lims[0],data_obj['H']<h_lims[1])]
        data_obj=data_obj[np.subtract(np.subtract(data_obj['J'],data_obj['K']),1.5*data_obj['AK_TARG'])>color_lim]

        # find center and annulus stars in object file
        center_stars_obj=stars_within_radius(data_obj,1*cluster.radius,cluster.center)[0]
        center_stars_obj=center_stars_obj[data_obj[center_stars_obj]['PMRA_ERR']<8]

        annulus_stars_obj=np.setdiff1d(stars_within_radius(data_obj,2*cluster.radius,cluster.center),
                                       stars_within_radius(data_obj,1.5*cluster.radius,cluster.center))
        annulus_stars_obj=annulus_stars_obj[data_obj[annulus_stars_obj]['PMRA_ERR']<8]
        
        # all stars in the sky area (2x radius)
        total_stars_obj=stars_within_radius(data_obj,2*cluster.radius,cluster.center)[0]
        # fixing the APOGEE_ID to resemble the 2MASS ID
        for i in range(len(data_obj)):
            data_obj['APOGEE_ID'][i]=data_obj['APOGEE_ID'][i][2:]
            data_obj['APOGEE_ID'][i]=data_obj['APOGEE_ID'][i].strip()

        data_df_annulus=Table(data_obj[annulus_stars_obj]).to_pandas()
        data_df_center=Table(data_obj[center_stars_obj]).to_pandas()
        data_df_total=Table(data_obj[total_stars_obj]).to_pandas()
        
        # Gaia TAP query to get all the Gaia stars in the sky area
        try:
            gaia_pm=pd.read_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_Gaia_TAP_file')
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
            gaia_pm.to_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_Gaia_TAP_file')

        # join APOGEE object file with Gaia TAP file
        merged_table_annulus=data_df_annulus.merge(gaia_pm,how='inner',left_on='APOGEE_ID',\
                                                   right_on='original_ext_source_id')

        merged_table_center=gaia_pm.merge(data_df_center,how='inner',left_on='original_ext_source_id',\
                                          right_on='APOGEE_ID')
        
        merged_table_total=gaia_pm.merge(data_df_total,how='inner',left_on='original_ext_source_id',\
                                          right_on='APOGEE_ID')

        merged_table_center.to_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_merged_center_file')
        merged_table_annulus.to_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_merged_annulus_file')
        merged_table_total.to_csv(data_directory+\
                'Membership_Pipeline_Files/'+cluster.name+'_merged_total_file')
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
    # main function called to determine cluster members
    try:
	# list of recovered clusters
        global recovered_cluster_list
	# define an object of the class 'Cluster'
        cluster=Cluster(cluster_name)
    
        #Finding center stars and annulus stars
        center_stars=stars_within_radius(data_new,1*cluster.radius,cluster.center)[0]
        annulus_stars=np.setdiff1d(stars_within_radius(data_new,2*cluster.radius,cluster.center),
                                   stars_within_radius(data_new,1.5*cluster.radius,cluster.center))
        total_stars=stars_within_radius(data_new,2*cluster.radius,cluster.center)[0]
        
        data_center_stars=data_new[center_stars]
        data_annulus_stars=data_new[annulus_stars]
        data_total_stars=data_new[total_stars]
        
        # see what we've got
        print(cluster_name,' Central stars=',len(center_stars),'; Annulus Stars=',len(annulus_stars),
               ' Total stars=',len(total_stars))

        #at least 5 stars in the central region
        if len(data_center_stars)>5:
            
            # find all different fields spanned by stars in this patch of sky
            uniq=np.unique(data_total_stars['FIELD'],return_counts=True)
            # color limits as seen in center stars in APOGEE
            color_lim=np.min(np.subtract(np.subtract(data_total_stars['J'],data_total_stars['K']),
                                         1.5*data_total_stars['AK_TARG']))
            
            # get Gaia center and annulus stars using Gaia TAP query
            phot_gaia_center, phot_gaia_annulus, phot_gaia_total = get_Gaia_PMs(cluster,uniq[0],\
                            [np.min(data_total_stars['H']), np.max(data_total_stars['H'])], color_lim)
            # extract PMs and RVs for the Gaia stars
            # making sure pmra and pmdec from Gaia do not have high errors
            pm_center_err_cond = [np.logical_and(phot_gaia_center['pmra_error']<2.0,\
                                                 phot_gaia_center['pmdec_error']<2.0)][0]
            pm_annulus_err_cond = [np.logical_and(phot_gaia_annulus['pmra_error']<2.0,\
                                                 phot_gaia_annulus['pmdec_error']<2.0)][0]
            PM_RA_center=np.array(phot_gaia_center['pmra'][pm_center_err_cond])
            PM_DEC_center=np.array(phot_gaia_center['pmdec'][pm_center_err_cond])
            PM_RA_annulus=np.array(phot_gaia_annulus['pmra'][pm_annulus_err_cond])
            PM_DEC_annulus=np.array(phot_gaia_annulus['pmdec'][pm_annulus_err_cond])
          
            
            # fit the PMs with 2D Gaussian
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
            
            # same for Radial Velocities
            annulus_dens_V_RAD = KDE(data_annulus_stars['VHELIO_AVG'],x,center_stars)            
            center_dens_V_RAD = KDE(data_center_stars['VHELIO_AVG'],x,center_stars)
            subtracted_dens_V_RAD=center_dens_V_RAD-annulus_dens_V_RAD#subtracting annulus distr from central
            subtracted_dens_V_RAD[subtracted_dens_V_RAD<0]=0
            
            parameters_V_RAD,covariances_V_RAD,x=Gaussian_fit(x,subtracted_dens_V_RAD,cluster_name)#fitting
            
            # quality of fit for Radial Velocity
            peak=np.array([np.argmax(subtracted_dens_V_RAD)])
            residual=np.std(subtracted_dens_V_RAD[np.where(np.logical_or(x<x[peak[0]]-6,x>x[peak[0]]+6))])
            Ampl_residual_RV=subtracted_dens_V_RAD[peak[0]]/residual
            peak_separation_RV=abs(parameters_V_RAD[0]-np.median(data_annulus_stars['VHELIO_AVG']))/\
                np.sqrt(parameters_V_RAD[1]**2+st.median_absolute_deviation(data_annulus_stars['VHELIO_AVG'])**2)
            
            # rescaling fit for RV
            fit=norm.pdf(x,parameters_V_RAD[0],parameters_V_RAD[1])
            fit_factor=subtracted_dens_V_RAD[peak[0]]/np.max(fit)
            fit=fit*fit_factor
            normed_fit=fit/max(fit)

            # find probabilities in each dimension
            prob_RV=np.zeros(len(data_total_stars))
            prob_PM=np.zeros(len(data_total_stars))
            dist_center=np.zeros(len(data_total_stars))

            for count in range(len(data_total_stars['VHELIO_AVG'])):
                prob_RV[count]=prob_member(data_total_stars['VHELIO_AVG'][count],parameters_V_RAD)#RV probability
                dist_center[count]=dist_center_function(list(data_total_stars['RA','DEC'][count]),
                                                        cluster.center)#distance from center of cluster
                i,j=data_total_stars['GAIA_PMRA'][count],data_total_stars['GAIA_PMDEC'][count]
                prob_PM[count]=prob_member_2D([i,j],parameters_PM)#PM probability in a separate function
            
            # calculate number of sigmas from probability
            n_sigmas_RV=n_sigma_function(prob_RV)
            n_sigmas_PM=n_sigma_function(prob_PM)
            
            # modify these parameters to change sensitivity to the kinematics dimensions while determining members
            w1=1#peak_separation_RV
            w2=1#peak_separation_PM
            A1=1
            A2=1
            prob_total=np.exp((A1*w1*np.log(prob_RV)+A2*w2*np.log(prob_PM))/(A1*w1+A2*w2))
            n_sigmas_total=n_sigma_function(prob_total)
            
	    # number of cluster stars found	
            data_new_cluster=data_total_stars[n_sigmas_total<3]    
            print('Number of cluster stars=',len(data_new_cluster))

            # define parameters to save in file
            M_H=np.mean(data_new_cluster["M_H"])#Complete_Clusters[Complete_Clusters['NAME']==cluster_name]["METALLICITY"].values
            M_H_dispersion=np.std(data_new_cluster["M_H"][data_new_cluster['M_H']>-9000])#Complete_Clusters[Complete_Clusters['NAME']==cluster_name]["METALLICITY_ERROR"].values
            num_bkgd_stars = len(data_annulus_stars)
            num_cluster_members = len(data_new_cluster)
            velocity_dispersion = np.std(data_new_cluster['VHELIO_AVG'])
	    # use flags to identify 'good' and 'bad' recoveries
            if (np.logical_and(len(data_new_cluster)>3,Ampl_residual_RV>9.5)):
                validation_flag = 'GOOD'
            else:
                validation_flag = 'BAD'

#             save file with total stars including center stars and annulus stars with their no. of sigmas
            ascii.write([[cluster.name]*len(data_total_stars),data_total_stars['APOGEE_ID'],
                         data_total_stars['RA'],data_total_stars['DEC'],data_total_stars['GLON'],
                        data_total_stars['GLAT'],n_sigmas_RV,n_sigmas_PM,n_sigmas_total,dist_center,
                         [cluster.log_age]*len(data_total_stars)],
            '/uufs/astro.utah.edu/common/home/u1063369/Documents/Membership Pipeline Run 8/Files/'+cluster_name+
                        '_members_and_background_Kharchenko.dat', 
                        names=['Cluster','APOGEE_ID','RA','DEC','GLON','GLAT','no_sigmas_RV','no_sigmas_PM', 
                               'no_sigmas_total','dist_center','log_age'])
#             save a separate file with cluster parameters
            ascii.write([[cluster.name],[cluster.status],[cluster.center.ra.degree],\
                         [cluster.center.dec.degree],[cluster.radius],[cluster.distance],[APOGEE_distance],\
                         [APOGEE_distance_err],[velocity_dispersion],[cluster.log_age],\
                         [cluster.log_age_err]\
                         ,[M_H],[M_H_dispersion],[parameters_V_RAD[0]],[parameters_V_RAD[1]],\
                         [parameters_V_RAD[2]],[parameters_PM[0]],[parameters_PM[1]],[parameters_PM[2]],\
                         [parameters_PM[3]],[parameters_PM[4]],[parameters_PM[5]],[parameters_PM[6]],\
                         [uniq[0][np.argmax(uniq[1])]],[Ampl_residual_RV],[peak_separation_RV],\
                         [peak_separation_PM],[num_bkgd_stars],[num_cluster_members],[validation_flag]],\
            '/uufs/astro.utah.edu/common/home/u1063369/Documents/Membership Pipeline Run 8/Files/'+cluster_name+
                        '_RV_and_PM_fit_parameters_Kharchenko.dat',
                names=list(['Cluster_Name','Cluster_Status','Cluster_Center_RA','Cluster_Center_DEC',
                'Cluster_Radius','Heliocentric_Distance_Catalog','Heliocentric_Distance_APOGEE',\
                'Heliocentric_Distance_err_APOGEE','Velocity_dispersion','log_age','log_age_err','Metallicity',
                'Metallicity_dispersion','RV_fit_Mean','RV_fit_Std_Dev','RV_fit_Amplitude','PM_fit_RA Mean',
                'PM_fit_DEC_Mean','PM_fit_RA_Std_Dev','PM_fit_DEC_Std_Dev','PM_fit_Amplitude','PM_fit_Theta',
                'PM_fit_Background_Scale','Field Used for PMs','Ampl_residual_RV','peak_separation_RV',\
                'peak_separation_PM','num_background_stars','num_cluster_members','Validation']))                
        
            n_sigmas_total=n_sigmas_total[n_sigmas_total<3]

	   # now let's make some plots of what we've recovered!
           # condition for number of cluster members/RV fit quality
           if (np.logical_and(len(data_new_cluster)>3,Ampl_residual_RV>9.5)): 
                               
                # quality bitmasks
                bitmask_cond = np.logical_and.reduce((\
                    data_new_cluster['STARFLAG'] & 2**17 == 0, data_new_cluster['STARFLAG'] & 2**2 == 0,\
                    data_new_cluster['STARFLAG'] & 2**3 == 0, data_new_cluster['ASPCAPFLAG'] & 2**19 == 0,\
                    data_new_cluster['ASPCAPFLAG'] & 2**20 == 0, data_new_cluster['ASPCAPFLAG'] & 2**23 == 0))

                data_new_cluster_bitmask = data_new_cluster[bitmask_cond]
                n_sigmas_total_bitmask = n_sigmas_total[bitmask_cond]

		# define figure parameters
                fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7,10))
                pm_range_x = [-10,5]
                pm_range_y = [-10,5]
                inner_color = 'blue'
                annulus_color = 'green'
                field_color = 'gray'

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

                #RV plot
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
                
                fig.savefig(data_directory+'Membership_Pipeline_Files/'+cluster_name+'.png',dpi=200)
                plt.show()
                                
                return 1.
            else:
                print('Number of cluster stars is less (UNRECOVERED)')
                return 0.
        else:
            print('Number of center stars is less')
            return 0.
    except Exception:
        print('Exception')
        return 0.
    except ValueError:
        print('ValueError')
        return 0.
    except RuntimeError:
        print('RuntimeError')
        return 0.
    except IndexError:
        print('IndexError')
        return 0.
    except IOError:
        print('IOError')
        return 0.
    except KeyError:
        print('KeyError')
        return 0.
    except TypeError:
        print('TypeError')
        return 0.
    except ZeroDivisionError:
        print('ZeroDivisionError')
        return 0.
    except :
        print('Unexpected Error')
        return 0.




