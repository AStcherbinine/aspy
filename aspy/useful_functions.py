#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## useful_functions.py
## Created by Aurélien STCHERBININE
## Last modified by Aurélien STCHERBININE : 14/08/2023

##-----------------------------------------------------------------------------------
"""Useful generics functions.
"""
##-----------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------
## Packages
# Global
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import curve_fit
import pickle
import datetime
import scipy.constants as const
import glob
import os
# Local
from . import constants_perso as const2


##-----------------------------------------------------------------------------------
## Sauvegarde figure()
def savefig3(fig, name, folder=''):
    """Save the matplotlip.pyplot figure as 3 differents files:
        - png image
        - svg image
        - pdf document

    The figure will be saved as `folder + name + {ext}`.

    Parameters
    ==========
    fig : Figure
        The matplotlib.pyplot figure to save.
    name : str
        The filename to save the figure (without format extension).
        ex: 'figure' -> figure.png / figure.pdf / figure.pkl
    folder : str, optional (default '')
        The target folder for saving.
    """
    basename = os.path.join(folder, name)
    fig.savefig(basename + '.png')
    fig.savefig(basename + '.svg')
    fig.savefig(basename + '.pdf')
    print('\033[01;34mFigure saved\033[0m')

def savefig4(fig, name, folder=''):
    """Save the matplotlip.pyplot figure as 4 differents files:
        - png image
        - svg image
        - pdf document
        - pickle file

    The figure will be saved as `folder + name + {ext}`.

    Parameters
    ==========
    fig : Figure
        The matplotlib.pyplot figure to save.
    name : str
        The filename to save the figure (without format extension).
        ex: 'figure' -> figure.png / figure.svg /figure.pdf / figure.pkl
    folder : str, optional (default '')
        The target folder for saving.
    """
    basename = os.path.join(folder, name)
    fig.savefig(basename + '.png')
    fig.savefig(basename + '.svg')
    fig.savefig(basename + '.pdf')
    with open(basename + '.pkl', 'wb') as output:
        pickle.dump(fig, output)
    print('\033[01;34mFigure saved\033[0m')

def load_fig(filename):
    """Load the plt.Figure previously saved as a pickle (*.pkl) object at filename.

    Parameters
    ==========
    filename : str
        The path of the figure pickle file to load.

    Returns
    =======
    fig : Figure
        The maptplotlib.pyplot figure loaded.
    """
    filename = myglob(filename)
    with open(filename, 'rb') as input_file:
        fig = pickle.load(input_file)
    print('\033[03m' + filename + '\033[0;01;34m loaded\033[0m')
    return fig

##-----------------------------------------------------------------------------------
## Ajustement
def f_lin(x, a, b):
    """Fonction linéaire : renvoie f(x) = a*x + b
    
    Parameters
    ==========
    x : float or ndarray
    a : float
        The line slope.
    b : float
        The origin ordinate.
    
    Returns
    =======
    f(x) = a*x + b : float or ndarray
    """
    return a*x + b

def reg_lin(X, Y, **kwargs):
    """Renvoie le résultat de la régression linéaire ( f(x) = a*x + b ) sur les valeurs 
    en entrée.
    
    Parameters
    ==========
    X : ndarray
        The X-values.
    Y : ndarray
        The Y-values.
    **kwargs
        Optional keyword arguments to pass to the scipy.optimize.curve_fit function.

    Returns
    =======
    a : float
        Slope of the fitted line.
    b : float
        Origin ordinate of the fitted line.
    """
    a, b = curve_fit(f_lin, X, Y, **kwargs)[0]
    return a, b

def planck(lam, T):
    """Return the Black body radiance, associated to the input wavelength and
    temperature. According to the Planck's law.

    Parameters
    ==========
    lam : float or array-like
        The wavelength (in m).
    T : float
        The temperature (in K).
    
    Returns
    =======
    B_lam : float or array-like
        The spectral radiance (in W.m-2.sr-1.m-1).
    """
    h = const.h
    c = const.c
    kB = const.k
    B_lam = (2*h*c*c) / (lam**5) / (np.exp(h*c / (lam*kB*T)) - 1)
    return B_lam

def fit_black_body(lam, sp, T_bounds=(0, 1e6)):
    """Return the temperature associated to the fitted black body thermical
    spectrum.

    Parameters
    ==========
    lam : array-like
        The wavelength array (in m).
    sp : array-like
        The spectral radiance (in W.m-2.sr-1.m-1) to be fitted.
    bounds : 2-tuple, optional (default (0, 1e6))
        The bounds for the temperature fitting.

    Returns
    =======
    T : float
        The temperature of the fitted Planck's law radiance (in K).
    """
    T = curve_fit(planck, lam, sp, bounds=T_bounds)[0][0]
    return T

def planck_inv(lam, B_lam):
    """Return the brightness temperature associated to the input wavelength and
    Black body radiance. According to the Planck's law.

    Parameters
    ==========
    lam : float
        The wavelength (in m).
    B_lam : float
        The spectral radiance (in W.m-2.sr-1.m-1).
    
    Returns
    =======
    T : float
        The temperature (in K).
    """
    h = const.h
    c = const.c
    kB = const.k
    T = (h*c) / (lam*kB) / np.log(1 + (2*h*c*c) / (lam**5 * B_lam))
    return T

def degre2(x, a, b, c):
    """Polynôme d'ordre 2.

    Parameters
    ==========
    x : array-like or float
    a : float
    b : float
    c : float

    Returns
    =======
    y : float
        y = a*x**2 + b*x + c
    """
    return a*x*x + b*x + c

def degre3(x, a, b, c, d):
    """Polynôme d'ordre 3.

    Parameters
    ==========
    x : array-like or float
    a : float
    b : float
    c : float
    d : float

    Returns
    =======
    y : float
        y = a*x**3 + b*x**2 + c*x + d
    """
    return a*x*x*x + b*x*x + c*x + d

##-----------------------------------------------------------------------------------
## Filtrage
def filtre_median(sp, n):
    """Applique un filtre médian sur les valeurs du spectre en remplaçant chaque valeur 
    par la médiane des valeurs dans un intervalle de dimension 2n+1 centré sur la valeur
    en question.

    Parameters
    ==========
    sp : ndarray
        Array of transmittance values.
    n : int
        The len of the window the moving median is 2n+1.

    Returns
    =======
    sp_med : ndarray
        Filtered transmittance array.
    """
    if n==0:
        return sp
    elif n < 0:
        raise ValueError('n must be >= 0')
    sp_med = deepcopy(sp)
    for i in range(n):
        if np.isnan(sp[i]):
            sp_med[i] = np.nan
        else:
            sp_med[i] = np.nanmedian(sp_med[:2*i+1])
    for i in range(n, len(sp)-n):
        if np.isnan(sp[i]):
            sp_med[i] = np.nan
        else:
            sp_med[i] = np.nanmedian(sp_med[i-n:i+n+1])
    for i in range(len(sp)-n, len(sp)):
        if np.isnan(sp[i]):
            sp_med[i] = np.nan
        else:
            sp_med[i] = np.nanmedian(sp_med[-2*(len(sp)-i):])
    return sp_med

def moyenne_glissante(sp, n):
    """Applique un filtre de moyenne glissante sur les valeurs du spectre en remplaçant 
    chaque valeur par la moyenne des valeurs dans un intervalle de dimension 2n+1 centré 
    sur la valeur en question.

    Parameters
    ==========
    sp : ndarray
        Array of the transmittance values.
    n : int
        The len of the window of the moving average is 2n+1.

    Returns
    =======
    sp_med : ndarray
        Filtered transmittance array.
    """
    if n==0:
        return sp
    elif n < 0:
        raise ValueError('n must be >= 0')
    sp_moy = deepcopy(sp)
    for i in range(n):
        if np.isnan(sp[i]):
            sp_moy[i] = np.nan
        else:
            sp_moy[i] = np.nanmean(sp_moy[:2*i+1])
    for i in range(n, len(sp)-n):
        if np.isnan(sp[i]):
            sp_moy[i] = np.nan
        else:
            sp_moy[i] = np.nanmean(sp_moy[i-n:i+n+1])
    for i in range(len(sp)-n, len(sp)):
        if np.isnan(sp[i]):
            sp_moy[i] = np.nan
        else:
            sp_moy[i] = np.nanmean(sp_moy[-2*(len(sp)-i):])
    return sp_moy

##-----------------------------------------------------------------------------------
## Recherche
def where_closer(value, array):
    """Renvoie l'indice de la valeur la plus proche de celle recherchée dans array.
    
    Parameters
    ==========
    values : float
        Searched value.
    array : ndarray
        The array.

    Returns
    =======
    i : int
        The index of the closer value to value in array.
    """
    array2 = np.abs(array - value)
    i_closer = np.where(array2 == np.nanmin(array2))[0][0]
    return i_closer

def where_closer_array(values, array):
    """Renvoie la liste des indices des valeurs les plus proches de celles recherchées
    dans array.

    Parameters
    ==========
    values : ndarray
        Array of searched values.
    array : ndarray
        The array.

    Returns
    =======
    I : ndarray
        Array of the index of the closer values in the array.
    """
    i_closer = []
    for val in values:
#        array2 = np.abs(array - val)
#        i_closer.append(np.where(array2 == array2.min())[0][0])
        i_closer.append(where_closer(val, array))
    return np.array(i_closer)

##----------------------------------------------------------------------------------------
## Conversion utc / loct -> datetime
def utc_to_datetime(utc):
    """Convert an ACSdata utc-like binary date to datetime.
    ex: b'2018OCT01 09:00'

    Parameters
    ==========
    utc : bytes
        ACSdata utc-like binary date.

    Returns
    =======
    utc_dt : datetime.datetime
        The date as a datetime.datetime object.
    """
    utc_dt = datetime.datetime.strptime(utc.decode('utf-8'), '%Y%b%d %H:%M')
    return utc_dt

def utc_array_to_datetime(utc):
    """Convert an array of ACSdata utc-like binary date to an array of datetime.
    ex: np.array([b'2018OCT01 09:00'])

    Parameters
    ==========
    utc : array of bytes
        ACSdata utc-like binary date ndarray.

    Returns
    =======
    utc_dt : array of datetime.datetime
        The array of the dates as datetime.datetime objects.
    """
    utc_dt = []
    for i in range(len(utc)):
        utc_b = utc[i]
        utc_dt.append(datetime.datetime.strptime(utc_b.decode('utf-8'), '%Y%b%d %H:%M'))
    return np.array(utc_dt)

def loct_to_datetime(loct):
    """Convert an ACSdata loct-like binary date to datetime.
    ex: b'09:00:00'

    Parameters
    ==========
    loct : bytes 
        ACSdata loct-like binary time.

    Returns
    =======
    loct_dt : datetime.time
        The local time as a datetime.time object.
    """
    loct_dt = datetime.datetime.strptime(loct.decode('utf-8')[:5], '%H:%M').time()
    return loct_dt

def loct_array_to_datetime(loct):
    """Convert an ACSdata loct-like binary date to datetime.
    ex: np.array([b'09:00:00'])

    Parameters
    ==========
    loct : bytes
        ACSdata loct-like binary time ndarray.

    Returns
    =======
    loct_dt : array of datetime.time
        The array of the local time as a datetime.time object.
    """
    loct_dt = []
    for i in range(len(loct)):
        loct_b = loct[i]
        loct_dt.append( datetime.datetime.strptime(loct_b.decode('utf-8')[:5], 
                                                                 '%H:%M').time() )
                            # on ne prend pas en compte les secondes s'il y en a
    return np.array(loct_dt)

def time_to_datetime(time, date=datetime.date(1995, 9, 3)):
    """Convert a datetime.time object to a datetime.datetime object,
    with adding the input datetime.date.

    Parameters
    ==========
    time : datetime.time
        The input time.
    date : datetime.date, optional (default datetime.date(1995, 9, 3))
        The input date.

    Returns
    =======
    datetime : datetime.datetime
        The combination of date and time as a datetime.datetime object.
    """
    return datetime.datetime.combine(date, time)

def time_to_datetime_array(time_array, date=datetime.date(1995, 9, 3)):
    """Convert an array of datetime.time objects to an other of 
    datetime.datetime object, with adding the input datetime.date.

    Parameters
    ==========
    time : array of datetime.time
        The input time array.
    date : datetime.date, optional (default datetime.date(1995, 9, 3))
        The input date.

    Returns
    =======
    datetime : array of datetime.datetime
        The array of combination of date and time as a datetime.datetime object.
    """
    datetime_array = deepcopy(time_array)
    for i in range(len(datetime_array)):
        datetime_array[i] = time_to_datetime(time_array[i], date)
    return datetime_array

def datetime_to_float(time):
    """Convert a datetime.time object to a float
    ex: 12h30min00s -> 12.5

    Parameters
    ==========
    time : datetime.time
        The time as a datetime.

    Returns
    =======
    float : float
        The time as a float.
    """
    return time.hour + time.minute/60 + time.second/3600

def datetime_to_float_array(time_array):
    """Convert an array of datetime.time objects to an other of floats.
    ex: 12h30min00s -> 12.5

    Parameters
    ==========
    time : array of datetime.time
        The array of time as datetime.

    Returns
    =======
    float : array of float
        The array of time as floats.
    """
    float_array = deepcopy(time_array)
    for i in range(len(time_array)):
        float_array[i] = datetime_to_float(time_array[i])
    return float_array

##-----------------------------------------------------------------------------------
## Recherche nom fichier
def myglob(basename, exclude=[]):
    """Return the absolute path according to the input basename.
    If mulitple files corresponds to the basename, the user will be asked
    to choose one.

    --------------------------------------------
    | int -> Select the corresponding filename.
    | q/quit/exit -> Return None.
    | a/all -> Return the list of all filenames.
    --------------------------------------------

    Parameters
    ==========
    basename : str
        The basename of the target file.
    exclude : list or np.ndarray of str, optional (default [])
        List of sub-strings to exclude from the results.

    Returns
    =======
    fname : str
        The absolute path of the selected file.
    """
    fnames = glob.glob(basename)
    if not isinstance(exclude, (list, np.ndarray)):
        raise ValueError("exclude parameter must be a list or numpy.ndarray")
    if len(exclude) > 0:
        fnames2 = []
        for name in fnames:
            test = True
            for excl in exclude:
                if excl in name:
                    test = False
                    continue
            if test:
                fnames2.append(name)
        fnames = fnames2
    fnames.sort()
    if fnames == []:
        # raise ValueError("No such file found.")
        print("\033[1;33mNo such file found.\033[0m")
        return None
    elif len(fnames) == 1:
        return fnames[0]
    else:
        dico = {}
        print('\033[1m{0} files found :\033[0m'.format(len(fnames)))
        for i, fname in enumerate(fnames):
            dico[str(i+1)] = fname
            print('{0:>2d} : \033[3m{1}\033[0m'.format(i+1, fname))
        print('\n\033[1mEnter the corresponding number to select one filename :\033[0m')
        while True:
            try:
                # n = input('Selection : ')
                n = input('>>> ')
                if n in dico.keys():
                    return dico[n]
                elif n=='q' or n=='quit' or n=='exit':
                    return None
                elif n=='a' or n=='all':
                    return fnames
                else:
                    print('Error, please enter an integer between 1 and ' 
                        + '{0}'.format(len(fnames)))
            except KeyboardInterrupt:
                return None

##-----------------------------------------------------------------------------------
## Tri
def sort_dict(dico):
    """Sort a dictionary by its keys values.

    Parameters
    ==========
    dico : dict
        The input unsorted dictionary.

    Returns
    =======
    dico_sorted : dict
        The sordet dictionary.
    """
    # Conversion en np.arrays
    values = np.array(list(dico.values()))
    keys = np.array(list(dico.keys()))
    # Tri par valeurs de clé croissantes
    i_ord = np.argsort(keys)
    keys2 = deepcopy(keys[i_ord])
    values2 = deepcopy(values[i_ord])
    # Sauvegarde dans un nouveau dictionnaire 
    dico_sorted = {}
    for i in range(len(keys2)):
        dico_sorted[keys2[i]] = values2[i]
    return dico_sorted

##-----------------------------------------------------------------------------------
## Sauvegarde / Importation
def save_pickle(obj, target_path, disp=True):
    """Save an object at the selected path using the pickle module.

    Parameters
    ==========
    obj : Object
        The object to save.
    target_path : str
        The saving path name.
    disp : bool
        Control the display.
            | True -> Print the saving filename.
            | False -> Nothing printed.
    """
    with open(target_path, 'wb') as output:
        pickle.dump(obj, output)
    if disp:
        print('\033[01;34mSaved as \033[0;03m' + target_path + '\033[0m')

def load_pickle(filename, disp=True):
    """Load and return a previously saved object with pickle.

    Parameters
    ==========
    filename : str
        The file path.
    disp : bool
        Control the display.
            | True -> Print the loading filename.
            | False -> Nothing printed.

    Returns
    =======
    obj : Object
        The loaded object.
    """
    filename2 = myglob(filename)
    with open(filename2, 'rb') as input_file:
        obj = pickle.load(input_file)
        if disp:
            print('\033[03m' + filename2 + '\033[0;01;34m loaded\033[0m')
        return obj

def save_spectra(lam, sp, filename, sp_infos='Saved spectrum', sav_folder='../data/OMEGA/sav_spectra/',
                 sub_folder=''):
    """Save a spectrum to a text file.

    Final filepath : `sav_foder / sub_folder / filename`

    Parameters
    ==========
    lam : array-like
        The wavelength of the spectrum to be saved.
    sp : array-lik
        The spectrum to be saved.
    filename : str
        The target filename for the saved file.
    sp_infos : src, optional (default 'Saved spectrum')
        Informations about the spectrum, that will be added in the file header.
    sav_folder : str, optional (default '../data/OMEGA/sav_spectra/')
        The target folder path for the saved file.
    sub_folder : str, optional (default '')
        The sub-folder where the saved file will be save.
    """
    header = (sp_infos + "\n"
            + "Computed on {0:s}\n\n".format(datetime.datetime.now().strftime('%d-%m-%Y %H:%M'))
            + "Wavelength [µm]  Reflectance"
                )
    sav_filename = os.path.join(sav_folder, sub_folder, filename)
    if test_security_overwrite(sav_filename):
        np.savetxt(sav_filename, np.transpose([lam, sp]), fmt=['%.5f', '%.5f'], header=header, 
                   comments='# ', encoding='utf8')

##-----------------------------------------------------------------------------------
## Test existence avant sauvegarde
def test_security_overwrite(path):
    """Test if a file already exists, and if yes ask the user if he wants to
    ovewrite it or not.

    Parameters
    ==========
    path : str
        The target file path.

    Returns
    =======
    overwrite : bool
        | True -> No existent file, or overwriting allowed.
        | False -> Existent file, no overwriting.
    """
    erase = 'n'
    if glob.glob(path) != []:
        try:
            erase = input('Do you really want to erase and replace \033[3m' + path +
                        '\033[0m ? (y/N) ')
        except KeyboardInterrupt:
            erase = 'n'
        if erase != 'y' :
            print("\033[1mFile preserved\033[0m")
            return False
        else:
            return True
    else:
        return True

##-----------------------------------------------------------------------------------
## Fonction similaires IDL
def idl_smooth_1D(X, w, NaN=False):
    """Return a copy of the input array X, smoothed with a boxcar average of the
    specified width w.

    Parameters
    ==========
    X : 1D ndarray
        Array to be smoothed.
    w : int
        The width of the smoothing window.
    NaN : bool, optional (default False)
        | If True : the NaN values within the array will be treated as missing data
            and will be replaced.
        | If False : the NaN values are keeped to NaN, and ignored for the smoothing.

    Returns
    =======
    X_smoothed : 1D ndarray
        Smoothed array.
    """
    if w==0:
        return X
    elif w < 0:
        raise ValueError('The smoothing window width w must be >= 0')
    w2 = int(w//2)
    X_smoothed = deepcopy(X)
    for i in range(w2, len(X)-w2):
        if np.isnan(X[i]) and (not NaN):
            X_smoothed[i] = np.nan
        else:
            X_smoothed[i] = np.nanmean(X[i-w2:i+w2+1])
    return X_smoothed

def idl_median_1D(X, w):
    """Return a copy of the input array X, smoothed with a boxcar median of the
    specified width w.

    Parameters
    ==========
    X : 1D ndarray
        Array to be smoothed.
    w : int
        The width of the smoothing window.

    Returns
    =======
    X_med : 1D ndarray
        Smoothed array.
    """
    if w==0:
        return X
    elif w < 0:
        raise ValueError('The smoothing window width w must be >= 0')
    w2 = int(w//2)
    X_med = deepcopy(X)
    for i in range(w2, len(X)-w2):
        if np.isnan(X[i]) and (not NaN):
            X_med[i] = np.nan
        else:
            X_med[i] = np.nanmedian(X[i-w2:i+w2+1])
    return X_med

def idl_spline(X, Y, T, sigma = 1.0):
    """ Performs a cubic spline interpolation.

    Parameters
    ----------
	X : ndarray
        The abcissa vector. Values MUST be monotonically increasing.

	Y : ndarray
        The vector of ordinate values corresponding to X.

	T : ndarray
        The vector of abcissae values for which the ordinate is
		desired. The values of T MUST be monotonically increasing.
    
	Sigma : float, default 1.0
        The amount of "tension" that is applied to the curve. The
		default value is 1.0. If sigma is close to 0, (e.g., .01),
		then effectively there is a cubic spline fit. If sigma
		is large, (e.g., greater than 10), then the fit will be like
		a polynomial interpolation.
    
    Returns
    -------
    spl : ndarray
	    Vector of interpolated ordinates.
	    Result(i) = value of the function at T(i).
    """
    n = min(len(X), len(Y))
    if n <= 2:
        print('X and Y must be arrays of 3 or more elements.')
    if sigma != 1.0:
        sigma = min(sigma, 0.001)
    yp = np.zeros(2*n)
    delx1 = X[1]-X[0]
    dx1 = (Y[1]-Y[0])/delx1
    nm1 = n-1
    nmp = n+1
    delx2 = X[2]-X[1]
    delx12 = X[2]-X[0]
    c1 = -(delx12+delx1)/(delx12*delx1)
    c2 = delx12/(delx1*delx2)
    c3 = -delx1/(delx12*delx2)
    slpp1 = c1*Y[0]+c2*Y[1]+c3*Y[2]
    deln = X[nm1]-X[nm1-1]
    delnm1 = X[nm1-1]-X[nm1-2]
    delnn = X[nm1]-X[nm1-2]
    c1 = (delnn+deln)/(delnn*deln)
    c2 = -delnn/(deln*delnm1)
    c3 = deln/(delnn*delnm1)
    slppn = c3*Y[nm1-2]+c2*Y[nm1-1]+c1*Y[nm1]
    sigmap = sigma*nm1/(X[nm1]-X[0])
    dels = sigmap*delx1
    exps = np.exp(dels)
    sinhs = 0.5*(exps-1/exps)
    sinhin = 1/(delx1*sinhs)
    diag1 = sinhin*(dels*0.5*(exps+1/exps)-sinhs)
    diagin = 1/diag1
    yp[0] = diagin*(dx1-slpp1)
    spdiag = sinhin*(sinhs-dels)
    yp[n] = diagin*spdiag
    delx2 = X[1:]-X[:-1]
    dx2 = (Y[1:]-Y[:-1])/delx2
    dels = sigmap*delx2
    exps = np.exp(dels)
    sinhs = 0.5*(exps-1/exps)
    sinhin = 1/(delx2*sinhs)
    diag2 = sinhin*(dels*(0.5*(exps+1/exps))-sinhs)
    diag2 = np.concatenate([np.array([0]), diag2[:-1]+diag2[1:]])
    dx2nm1 = dx2[nm1-1]
    dx2 = np.concatenate([np.array([0]), dx2[1:]-dx2[:-1]])
    spdiag = sinhin*(sinhs-dels)
    for i in range(1, nm1):
        diagin = 1/(diag2[i]-spdiag[i-1]*yp[i+n-1])
        yp[i] = diagin*(dx2[i]-spdiag[i-1]*yp[i-1])
        yp[i+n] = diagin*spdiag[i]
    diagin = 1/(diag1-spdiag[nm1-1]*yp[n+nm1-1])
    yp[nm1] = diagin*(slppn-dx2nm1-spdiag[nm1-1]*yp[nm1-1])
    for i in range(n-2, -1, -1):
        yp[i] = yp[i]-yp[i+n]*yp[i+1]
    m = len(T)
    subs = np.repeat(nm1, m)
    s = X[nm1]-X[0]
    sigmap = sigma*nm1/s
    j = 0
    for i in range(1, nm1+1):
        while T[j] < X[i]:
            subs[j] = i
            j += 1
            if j == m:
                break
        if j == m:
            break
    subs1 = subs-1
    del1 = T-X[subs1]
    del2 = X[subs]-T
    dels = X[subs]-X[subs1]
    exps1 = np.exp(sigmap*del1)
    sinhd1 = 0.5*(exps1-1/exps1)
    exps = np.exp(sigmap*del2)
    sinhd2 = 0.5*(exps-1/exps)
    exps = exps1*exps
    sinhs = 0.5*(exps-1/exps)
    spl = (yp[subs]*sinhd1+yp[subs1]*sinhd2)/sinhs+((Y[subs]-yp[subs])*del1+(Y[subs1]-yp[subs1])*del2)/dels
    if m == 1:
        return spl[0]
    else:
        return spl

##-----------------------------------------------------------------------------------
## End of code
##-----------------------------------------------------------------------------------
