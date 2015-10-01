# Copyright 2015 John Walk
# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Toolkit for fitting non-powerlaw (NPL) scalings derived by Murari 
et al. to datasets.

Provides the functions:
(1) hfactor
    defines the saturation term in field and density
(2) NPL
    defines the non-powerlaw fit with arbitrary coefficients
(3) errfunct
    residuals of NPL
(4) fit_model
    least-squares fitter using errfunct
"""

import numpy as np
from scipy.optimize import leastsq

def hfactor(hparams,n,B):
    """suppression term used in NPL

    ARGS:
        hparams: list.
            list of parameter values for h-factor.
        n: array-like or float.
            plasma average density [10^19 m^-3]
        B: array-like or float.
            axial toroidal field [T]

    RETURNS:
        h: array-like or float.
            suppression factor.
    """
    return n**hparams[0] * (1. + np.exp(hparams[1] * (n/B)**hparams[2]))**-1.

def NPL(params,Ip,R,kappa,P,n,B):
    """non-powerlaw fit from Murari et al. of the form

    NPL = C * Ip^a1 * R^a2 * kappa^a3 * P^a4 * h(n,B)
    h(n,B) = n^a5 * (1 + exp(a6 * (n/B)^a7))^-1

    where h(n,B) is a saturation term introduced to break the powerlaw 
    symmetry.

    ARGS:
        params: list.
            list of parameter values for the model, [C,a1,a2,a3,a4,a5,a6,a7]
        Ip: array-like or float.
            plasma current [MA]
        R: array-like or float.
            plasma major radius [m]
        kappa: array-like or float.
            plasma elongation (dimensionless)
        P: array-like or float.
            heating power [MW]
        n: array-like or float.
            plasma average density [10^19 m^-3]
        B: array-like or float.
            axial toroidal field [T]

    RETURNS:
        tauE: array-like or float.
            calculated energy confinement time [s]
    """
    hparams = params[5:]
    h = hfactor(hparams,n,B)

    tauE = params[0] * Ip**params[1] * R**params[2] * kappa**params[3] * P**params[4] * h
    return tauE


def errfunct(params,Ip,R,kappa,P,n,B,tau):
    """error function minimized by leastsq using NPL.

    ARGS:
        params: list.
            list of parameter values for the model, [C,a1,a2,a3,a4,a5,a6,a7]
        Ip: array-like or float.
            plasma current [MA]
        R: array-like or float.
            plasma major radius [m]
        kappa: array-like or float.
            plasma elongation (dimensionless)
        P: array-like or float.
            heating power [MW]
        n: array-like or float.
            plasma average density (10^19 m^-3)
        B: array-like or float.
            axial toroidal field [T]
        tau: array-like or float.
            measured energy confinement time [s]

    RETURNS:
        resid: array-like or float.
            residuals of tau versus model.
    """
    resid = tau - NPL(params,Ip,R,kappa,P,n,B)
    return resid


def fit_model(guesses,Ip,R,kappa,P,n,B,values):
    """generates least-squares minimized model for given values modeled with NPL function.

    ARGS:
        guesses: list.
            initial guesses of the parameters for the NPL function.
        Ip: array-like or float.
            plasma current [MA]
        R: array-like or float.
            plasma major radius [m]
        kappa: array-like or float.
            plasma elongation (dimensionless)
        P: array-like or float.
            heating power [MW]
        n: array-like or float.
            plasma average density (10^19 m^-3)
        B: array-like or float.
            axial toroidal field [T]
        values: array-like or float.
            measured energy confinement time [s] (to be fitted)

    RETURNS:
        p1: list.
            least-squares optimized model parameters.
        err: vector.
            1-sigma errorbars for parameters.
        r2: float.
            R-squared coefficient of determination.
        cov: vector.
            covariance matrix from least-squares model.
    """
    args = (Ip,R,kappa,P,n,B,values)
    p1,cov,infodict,mesg,ier = leastsq(errfunct,guesses,args=args,full_output=True)

    # calculate R^2 value
    ss_err = (infodict['fvec']**2).sum()
    ss_tot = ((np.log10(values) - np.log10(values.mean()))**2).sum()
    r2 = 1. - (ss_err/ss_tot)

    if cov is None:
        n = len(p1)
        cov = np.zeros((n,n))

    # calculate errors of parameter estimates
    ss_err_wt = ss_err/(len(values) - len(guesses))
    cov_wt = cov * ss_err_wt
    errors = []
    for i in range(len(p1)):
        try:
            errors.append(np.absolute(cov_wt[i][i])**0.5)
        except:
            errors.append(0.0)
    errors = np.array(errors)

    return (p1,errors,r2,cov)





