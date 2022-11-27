#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import shutil
import glob
import gzip
import sys

import numpy as np
import netCDF4 as nc4
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import cmocean
import numpy.ma as ma

import cftime
import time
import pyart

import fnmatch
import os

            
from datetime import date

from ipyfilechooser import FileChooser


def scanco(DS):
    
    rangekm = DS['range'][:]/1000.0;
    elev = DS['elevation'][:];
    azim = DS['azimuth'][:];

    direction = np.sign(np.sin(np.mean(azim[:])*np.pi/180));
    
    r_earth = 6371;
    r_earth = r_earth*4/3;

    r = np.cos(elev[:,None]*np.pi/180) * rangekm[None,:];

    z = np.sin(elev[:,None]*np.pi/180) * rangekm[None,:] + np.sqrt(r**2 + r_earth**2) - r_earth;
    x = np.sin(azim[:,None]*np.pi/180) * np.cos(elev[:,None]*np.pi/180) * rangekm[None,:];
    y = np.cos(azim[:,None]*np.pi/180) * np.cos(elev[:,None]*np.pi/180) * rangekm[None,:];

    return direction,r,x,y,z

def rawplot(ncfile):
    
    DS = nc4.Dataset(ncfile);

    dtime0 = cftime.num2pydate(DS['time'][:],DS['time'].units)

    title_date = dtime0[0].strftime("%Y%m%d %H:%M UTC");
    figdate = dtime0[0].strftime("%Y%m%d%H%M%S")

    direction,r,x,y,z = scanco(DS);
    
    fig, ax = plt.subplots(2,1,figsize=(12,4*2),constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,wspace=0.2)

    h0=ax[0].pcolor(r,z,DS['ZED_H'][:-1,:-1],cmap='pyart_HomeyerRainbow',vmin=Zscale[0],vmax=Zscale[1])
    ax[0].set_ylim(hscale[0],hscale[1]);
    ax[0].set_xlim(dcscale[0],dcscale[1])
    cb0 = plt.colorbar(h0,ax=ax[0],orientation='horizontal',shrink=0.8);
    cb0.ax.set_xlabel("Reflectivity Factor (dBZ)");
    if direction<0:
        ax[0].invert_xaxis();

    h1=ax[1].pcolor(r,z,DS['VEL_HV'][:-1,:-1],cmap=cmocean.cm.balance,vmin=Vscale[0],vmax=Vscale[1])
    ax[1].set_ylim(hscale[0],hscale[1]);
    ax[1].set_xlim(dcscale[0],dcscale[1]);
    if direction<0:
        ax[1].invert_xaxis();

    cb1 = plt.colorbar(h1,ax=ax[1],orientation='horizontal',shrink=0.8);
    cb1.ax.set_xlabel("Doppler velocity (m/s)");
    
    DS.close();
    
def plot_rhi(ncfile):
  
  DS = nc4.Dataset(ncfile);
  # OFFSETS/CALIBRATIONS:

  dtime0 = cftime.num2pydate(DS['time'][:],DS['time'].units)

  title_date = dtime0[0].strftime("%Y%m%d %H:%M UTC");
  figdate = dtime0[0].strftime("%Y%m%d%H%M%S")

  rangeOFFSET=0 #-840 %m
  ZhOFFSET=0 #+7 %dB
  ZDROFFSET=0 #+0.6 %+0.6 %dB 

  SNR_threshold_HH=3.5;  	# for copolar H - default
  SNR_threshold_VV=3.5; 	# for copolar V - default

  SNR_threshold_HH=5;  	# for copolar H
  SNR_threshold_VV=5; 	# for copolar V
  SNR_threshold_X=3.5; 	# cross polar SNR threshold
  SNR_threshold_CXC=100; 	# rho_hv and L : 200 corresponds to bias of 0.995
  SNR_threshold_SPW=6; 	# spectral width thresholding
 
  oldrange = DS['range'][:];
  newrange = oldrange+rangeOFFSET;

  #newrange(range<0) = nan;


  Zh = DS['ZED_H'][:,:];    # Horizontal polarised reflectivity [dBZ] (copolar)
  Zv = Zh - DS['ZDR'][:,:]; # Vertically polarised reflectivity [dBZ] (copolar)
  Zx = Zh + DS['LDR'][:,:]; # cross polar V [dBZ]
  
  PDP    = DS['PDP'][:,:] + 0; #-5.8; % Differential Phase Shift (deg)

  SPW_HV = DS['SPW_HV'][:,:]; # spectral width using H & V pulses
  V      = DS['VEL_HV'][:,:]; # Velocity from both H&V pulses (max range +/- 14.86m/s)
  CXC    = DS['CXC'][:,:];    # copolar cross correlation rho_hv^2

  linZ  = 10.0**(Zh/10.); signalpower=linZ*0; # linZ = reflectivities in linear units [mm^6/m^3]
  linZv = 10.0**(Zv/10.); signalpowerv=linZv*0; 
  linZx = 10.0**(Zx/10.); 

  signalpowerx = linZx*0; 
  signalpower  = Zh; 
  signalpowerv=signalpower; 
  signalpowerx=signalpower; # Just to get right array dimensions

  signalpower  = linZ/oldrange[None,:]**2;
  signalpowerv = linZv/oldrange[None,:]**2;
  signalpowerx = linZx/oldrange[None,:]**2;

  direction,r,x,y,z = scanco(DS)

  #noise=5.347e-5; % estimated for 20110318
  #noisev=5.216e-5;
  #noisex=8.17e-5;
  #sdnoise=2.5423e-06; % estimate 7 june 2016
  #sdnoisex=3.4e-06; % --- %

  print(z.shape)
  emptygates=np.where(z>12); # & isnan(signalpower)==0  & isnan(signalpowerv)==0  & isnan(signalpowerx)==0
  print(emptygates);

  print(np.shape(emptygates));

  #if length(emptygates)>50
  noise=np.mean(signalpower[emptygates]); noisev=np.mean(signalpowerv[emptygates]); noisex=np.mean(signalpowerx[emptygates]);
  sdnoise=np.std(signalpower[emptygates]); sdnoisev=np.std(signalpowerv[emptygates]); sdnoisex=np.std(signalpowerx[emptygates]);
  #else % otherwise have to guess, based on previous data - this is most likely for PPIs
  #noise=2.5971e-10; noisev=2.5280e-10; noisex=3.3132e-11;
  #sdnoise=1.7784e-11; sdnoisev=1.7064e-11; sdnoisex=1.347e-12;
  #end
  
  print(noise)

  signalpower=signalpower-noise;	# subtract noise from signal
  signalpowerv=signalpowerv-noisev;
  signalpowerx=signalpowerx-noisex;

  SNRperpulseH=signalpower/noise; SNRperpulseV=signalpowerv/noisev; iii=np.where(SNRperpulseH<0); SNRperpulseH[iii]=0; iii=np.where(SNRperpulseV<0); SNRperpulseV[iii]=0;
  bringi_factor=np.sqrt((1+1./SNRperpulseH)*(1+1./SNRperpulseV));

  signalpower = ma.masked_where(signalpower<SNR_threshold_HH*sdnoise, signalpower); 
  signapowerv = ma.masked_where(signalpowerv<SNR_threshold_VV*sdnoise, signalpowerv); 
  signalpowerx = ma.masked_where(signalpowerx<SNR_threshold_X*sdnoisex, signalpowerx);

  linZ  = signalpower*newrange[None,:]**2;	# calculate linear Z from signal power using CORRECTED range
  linZv = signalpowerv*newrange[None,:]**2;
  linZx = signalpowerx*newrange[None,:]**2;

  Zh=10*ma.log10(linZ)+ZhOFFSET;			#	 ADD ON CALIBRATION FACTORS
  Zv=10*ma.log10(linZv)+ZhOFFSET-ZDROFFSET;
  ZDR=Zh-Zv;  # CALCULATE ZDR [dB]
  Zx=10*ma.log10(linZx)+ZhOFFSET-ZDROFFSET;
  LDR=Zx-Zh;	

  CXC=ma.masked_where(SNRperpulseH<SNR_threshold_CXC, CXC); 
  SPW_HV = ma.masked_where(signalpower<SNR_threshold_SPW*noise,SPW_HV); 
  #SPW_HV = ma.masked_where(index=find(isnan(signalpower))==1; SPW_HV(index)=nan; 

  CXC = ma.masked_where(CXC<=0, CXC);
  RHO_HV=ma.power(CXC,0.5);
  L=-ma.log10(1-RHO_HV);

  V = ma.masked_where(signalpower<SNR_threshold_HH*sdnoise, V); 
  PDP = ma.masked_where(signalpower<SNR_threshold_HH*sdnoise, PDP);


  # identify likely clutter:
  clutter=Zh.copy();
  clutter[~Zh.mask]=0;
  clutter[ZDR<-3] = 1;
  clutter[LDR>-10]=1;
  clutter[CXC<0.5]=1;
  clutter[PDP<-30]=1;
  clutter[np.logical_and(abs(V)<0.15,clutter==0)] = 0.5;
  clutter = ma.masked_where(signalpower<SNR_threshold_HH*sdnoise, clutter)


  from mpl_axes_aligner import align
  km2kft = 1.0/0.3048;

  direction,r,x,y,z = scanco(DS);

  print(direction)

  cmap = plt.cm.get_cmap('pyart_HomeyerRainbow', 6) 
  #cmap = plt.cm.get_cmap('pyart_LangRainbow12', 6) 

  #cmap = plt.cm.jet  # define the colormap
  # extract all colors from the .jet map
  cmaplist = [cmap(i) for i in range(cmap.N)]
  # force the first color entry to be grey
  cmaplist[-1] = (112/256.,12/256.,179/256., 1.)

  # create the new map
  cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)


  az = DS['azimuth'][0];

  fig, ax = plt.subplots(1,1,figsize=(12,4),constrained_layout=True)
  fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,wspace=0.2)

  h0=ax.pcolor(r,z,Zh[:-1,:-1],cmap=cmap,vmin=Zscale[0],vmax=Zscale[1])
  ax.set_ylim(hscale[0],hscale[1]);
  ax.set_xlim(dcscale[0],dcscale[1]);
  if direction<0:
    ax.invert_xaxis();
    
  ax0 = ax.twinx();
  ax0.set_ylim(hscale[0]*km2kft,hscale[1]*km2kft)
  ax0.grid(True);

  cb0 = plt.colorbar(h0,ax=ax,orientation='horizontal',shrink=0.8);
  cb0.ax.set_xlabel("Reflectivity Factor (dBZ)");

  ax.set_xlabel('Distance from Chilbolton [km]')
  ax.set_ylabel('Height [km]')
  ax0.set_ylabel('Height [kft]')
  figtitle = '{} Reflectivity [dBZ] RHI Az {:.2f}'.format(title_date,az)
  ax.set_title(figtitle)

  figname = 'camra_rhi_{}_AZ{:.2f}_ZED_H.png'.format(figdate,az)

  plt.savefig(os.path.join(figpath,figname),dpi=150)

  DS.close();

  def plot_ppi(ncfile):
    
    direction,r,x,y,z = scanco(DS);

    fig, ax = plt.subplots(1,1,figsize=(12,12),constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,wspace=0.2)

    h0=ax.pcolor(x,y,DS['ZED_H'][:-1,:-1],cmap='pyart_HomeyerRainbow',vmin=Zscale[0],vmax=Zscale[1])
    # ax.set_ylim(hscale[0],hscale[1]);
    # ax.set_xlim(dcscale[0],dcscale[1]);

    el = DS['elevation'][0];

    cb0 = plt.colorbar(h0,ax=ax,orientation='horizontal',shrink=0.8);
    cb0.ax.set_xlabel("Reflectivity Factor (dBZ)");

    ax.set_xlabel('Distance East from Chilbolton [km]')
    ax.set_ylabel('Distance North from Chilbolton [km]')
    figtitle = '{} Reflectivity [dBZ] PPI El {:.2f}'.format(title_date,el)
    ax.set_title(figtitle)
    ax.axis('equal')

    figname = 'camra_ppi_{}_ZED_H.png'.format(figdate)
    plt.savefig(os.path.join(figpath,figname),dpi=75)

    DS.close();

#print(np.sin(np.mean(DS['azimuth'][:])*np.pi/180))



today = date.today()
datestr = today.strftime("%Y%m%d")
# Uncomment the following line if you want to set the date manually
datestr = '20220908';


# Set scales on plots
dcscale  = [0   , 200];
hscale   = [0   , 12];
Zscale   = [-10 , 50];
ZDRscale = [-1  , 3];
Lscale   = [1   , 2.5];
PDPscale = [0   , 20];
Vscale   = [-15 , 15];
SPWscale = [0   , 2.5];
LDRscale = [-35 , -15];

print(dcscale,hscale);


camra_raw_path = '/focus/radar-camra/raw/{}'.format(datestr);

figpath = '/home/cw66/public_html/wescon'
figpath = os.path.join(figpath,datestr);
# Check whether the specified path exists or not
isExist = os.path.exists(figpath);

if not isExist:
  os.makedirs(figpath)

# Create and display a FileChooser widget
#fc = FileChooser(title="Select radar file",path=camra_raw_path)
#fc.filter_pattern = '*.nc'
#ncfile = fc.selected_filename;
#display(fc)

os.chdir(camra_raw_path);
ppis = [os.path.join(camra_raw_path,f) for f in glob.glob('*{}*ppi-raw.nc'.format(datestr))];
ppis.sort();
rhis = [os.path.join(camra_raw_path,f) for f in glob.glob('*{}*rhi-raw.nc'.format(datestr))];
rhis.sort();
mans = [os.path.join(camra_raw_path,f) for f in glob.glob('*{}*man-raw.nc'.format(datestr))];
mans.sort();

savedSet=set()

nameSet=set()
for file in os.listdir(camra_raw_path):
    fullpath=os.path.join(camra_raw_path, file)
    if os.path.isfile(fullpath):
        nameSet.add(file)
        
retrievedSet=set()
for name in nameSet:
    stat=os.stat(os.path.join(camra_raw_path, name))
    #time=ST_CTIME
    #size=stat.ST_SIZE If you add this, you will be able to detect file size changes as well.
    #Also consider using ST_MTIME to detect last time modified
    retrievedSet.add(name)
    
newSet=retrievedSet-savedSet

deletedSet=savedSet-retrievedSet

savedSet=newSet
newlist = list(newSet);
newlist.sort()


#rawplot(newlist[-1])
#for file in newSet:
#    print(file);
#    rawplot(file);




seconds_ago = 60*6;

files = (fle for rt, _, f in os.walk(camra_raw_path) for fle in f if time.time() - os.stat(
    os.path.join(rt, fle)).st_mtime < seconds_ago)

#print(list(files))


rhis = [os.path.join(camra_raw_path,f) for f in files if fnmatch.fnmatch(f,'*rhi-raw.nc')];

for f in rhis:
  print(f);
  plot_rhi(f);
  
#ncfile = os.path.join(camra_raw_path,fc.selected);

