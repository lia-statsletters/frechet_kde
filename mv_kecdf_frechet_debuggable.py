#from __future__ import division

from profilehooks import profile#, timecall
import cProfile
import pstats
import io


import pandas as pd
import numpy as np

import scipy.stats as spst
import statsmodels.api as sm
from scipy import optimize

import re
import time

import matplotlib.pyplot as plt

from statsmodels.nonparametric import kernels

kernel_func = dict(wangryzin=kernels.wang_ryzin,
                   aitchisonaitken=kernels.aitchison_aitken,
                   gaussian=kernels.gaussian,
                   aitchison_aitken_reg = kernels.aitchison_aitken_reg,
                   wangryzin_reg = kernels.wang_ryzin_reg,
                   gauss_convolution=kernels.gaussian_convolution,
                   wangryzin_convolution=kernels.wang_ryzin_convolution,
                   aitchisonaitken_convolution=kernels.aitchison_aitken_convolution,
                   gaussian_cdf=kernels.gaussian_cdf,
                   aitchisonaitken_cdf=kernels.aitchison_aitken_cdf,
                   wangryzin_cdf=kernels.wang_ryzin_cdf,
                   d_gaussian=kernels.d_gaussian)

def lets_be_tidy(rd,frechets):
    v_type = f'{"c"*len(rd)}'
    #threshold: number of violations by the cheapest method.
    dens_u_rot = sm.nonparametric.KDEMultivariate(data=rd, var_type=v_type, bw='normal_reference')
    cdf_dens_u_rot = dens_u_rot.cdf()
    violations_rot = count_frechet_fails(cdf_dens_u_rot, frechets)
    #how can we best use this violations?
    #nviolations= np.sum([np.sum(violations_rot['top']),
    #                     np.sum(violations_rot['bottom'])])

    tst = get_bw(rd, v_type, dens_u_rot.bw, frech_bounds=frechets)  # dens_u_rot.bw)

    #for comparison, call the package and check features and time
    dens_u_ml = sm.nonparametric.KDEMultivariate(data=rd,var_type=v_type, bw='cv_ml')

    print(tst, '\n', dens_u_rot.bw, '\n', dens_u_ml.bw)

def gpke(bwp, dataxx, data_predict, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    r"""Returns the non-normalized Generalized Product Kernel Estimator"""
    kertypes = dict(c=ckertype, o=okertype, u=ukertype)
    Kval = np.empty(dataxx.shape)
    for ii, vtype in enumerate(var_type):
        func = kernel_func[kertypes[vtype]]
        Kval[:, ii] = func(bwp[ii], dataxx[:, ii], data_predict[ii])

    iscontinuous = np.array([c == 'c' for c in var_type])
    dens = Kval.prod(axis=1) / np.prod(bwp[iscontinuous])
    #dens = np.nanprod(Kval,axis=1) / np.prod(bwp[iscontinuous])
    if tosum:
        return dens.sum(axis=0)
    else:
        return dens

def adjust_shape(dat, k_vars):
    """ Returns an array of shape (nobs, k_vars) for use with `gpke`."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and k_vars > 1:  # one obs many vars
        nobs = 1
    elif dat.ndim == 1 and k_vars == 1:  # one obs one var
        nobs = len(dat)
    else:
        if np.shape(dat)[0] == k_vars and np.shape(dat)[1] != k_vars:
            dat = dat.T

        nobs = np.shape(dat)[0]  # ndim >1 so many obs many vars

    dat = np.reshape(dat, (nobs, k_vars))
    return dat

def calc_frechet_fails(guinea_cdf,frechets):
    #fails = {'top': [], 'bottom': []}
    N=len(guinea_cdf)
    top=np.full(N,0.)
    bot=np.full(N,0.)
    for n in range(N):
        # n_hyper_point=np.array([x[n] for x in rd])
        xdiff=guinea_cdf[n]-frechets['top'][n]
        if xdiff>0:
            top[n]=xdiff

        xdiff=frechets['bottom'][n] - guinea_cdf[n]
        if xdiff>0:
            bot[n]=xdiff

    return {'top': top,
            'bottom': bot}


def count_frechet_fails(guinea_cdf,frechets,methodstring=''):
    fails={f'top_{methodstring}':[], f'bottom_{methodstring}':[]}
    for n in range(len(guinea_cdf)):
        amount=guinea_cdf[n]-frechets[f'top'][n]
        if amount>0:#guinea_cdf[n]>frechets['top'][n]:
            fails[f'top_{methodstring}'].append(amount)
            fails[f'bottom_{methodstring}'].append(0) #failed on top, cant fail bottom
        else:
            fails[f'top_{methodstring}'].append(0)
            #no fails in top, check for fails in bottom.
            amount=frechets[f'bottom'][n]-guinea_cdf[n]
            if amount>0:
                fails[f'bottom_{methodstring}'].append(amount)
            else:
                fails[f'bottom_{methodstring}'].append(0) #no fails on bottom either
    return {f'top_{methodstring}':np.array(fails[f'top_{methodstring}']),
            f'bottom_{methodstring}':np.array(fails[f'bottom_{methodstring}'])}

def get_frechets(F):
    #F is a list of "d" cdfs
    #Taken from remark 7.9 p225 of QRM book (McNeil, Frey, Embrechs)
    #For a multivariate df F with margins F1, ... Fd holds:
    # bottom <= F(\bf{x}) <= top ->
    # max( sum^{i=1}_{d} (F_i(x_i) ) + 1-d, 0 ) <= F(\bf{x}) <= min (F_1(x_1), ... F_d(x_d))

    d=len(F)
    samples=len(F[0])
    constd=1-d
    #easier-to-debug, arguably faster calculations
    bottom_frechet=np.full(samples,float('nan'))
    top_frechet=np.full(samples, float('nan'))
    for n in range(samples):
        bottom_frechet[n]=max(np.sum(F[:,n])+constd,0)
        top_frechet[n]=min(F[:,n])

    #sanity check: is bottom always lteq top?
    print('is bottom always lteq top?',
          sum(bottom_frechet<=top_frechet)==samples)

    return {'top': top_frechet, 'bottom': bottom_frechet}

class LeaveOneOut(object):
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        nobs, k_vars = np.shape(X)

        for i in range(nobs):
            index = np.ones(nobs, dtype=np.bool)
            index[i] = False
            yield X[index, :]


def cdf(dataxx, bw, var_type, frech_bounds=None):
    data_predict = dataxx
    nobs=np.shape(data_predict)[0]
    cdf_est = []
    #longer code but faster evaluation
    def ze_cdf_eval(bw, data_predict, i, var_type):
        return gpke(bw, data_predict, data_predict[i, :],
                    var_type,ckertype="gaussian_cdf",
                    ukertype="aitchisonaitken_cdf",okertype='wangryzin_cdf')

    #if not frech_bounds:
    for i in range(nobs):#np.shape(data_predict)[0]):
        ze_value=ze_cdf_eval(bw, data_predict, i, var_type)
        cdf_est.append( ze_value )
    cdf_est = np.squeeze(cdf_est)/ nobs
    return cdf_est

def frechet_likelihood(bww, datax, var_type, frech_bounds, func=None, debug_mode=False,):
    #Todo: REDO THIS FUNCTION SINCE THE VIOLATIONS ARE NOW DISTANCES!!!!!
    cdf_est = cdf(datax, bww, var_type)  # frech_bounds=frech_bounds)
    d_violations = calc_frechet_fails(cdf_est, frech_bounds)
    width_bound = frech_bounds['top'] - frech_bounds['bottom']
    viols=(d_violations['top']+d_violations['bottom'])/width_bound
    L= np.sum(viols)

    if debug_mode:
        nobs = len(datax)
        print(bww, 'violations (top,bottom):',
          f'({np.sum(~np.isin(d_violations["top"],0))},'
          f'{np.sum(~np.isin(d_violations["bottom"],0))})\n',
          'out of',nobs, 'samples, likelihood:', L, '\n')

    return L


def loo_likelihood(bww, datax, var_type, func=lambda x: x, debug_mode=False):
    #if frechet bounds available, check violations for this bandwidth.

    LOO = LeaveOneOut(datax) #iterable, one sample less for each sample.
    L = 0 #score
    for i, X_not_i in enumerate(LOO):
        f_i = gpke(bww, dataxx=-X_not_i, data_predict=-datax[i, :],
                   var_type=var_type)
        try:
            L += func(f_i)
        except:
            print('wtf f_i fucks up log likelihood in LOO!!! keep last value')
            continue

    if debug_mode:
        print('\n',bww,'Log likelihood:',-L)
    return -L


def get_bw(datapfft, var_type, reference, frech_bounds=None):
    # Using leave-one-out likelihood
    # the initial value for the optimization is the normal_reference
    # h0 = normal_reference()

    data = adjust_shape(datapfft, len(var_type))

    if not frech_bounds:
        fmin =lambda bw, funcx: loo_likelihood(bw, data, var_type, func=funcx)
        argsx=(np.log,)
    else:
        fmin = lambda bw, funcx: frechet_likelihood(bw, data, var_type,
                                                    frech_bounds, func=funcx)
        argsx=(None,) #second element of tuple is if debug mode

    h0 = reference
    #old
    #bw = optimize.fmin(fmin, x0=h0, args=argsx, #feeding logarithm for loo
    #                   maxiter=1e3, maxfun=1e3, disp=0, xtol=1e-3)
    #Gradient norm must be less than gtol before successful termination.
    bw = optimize.minimize(fmin,h0,method='BFGS',args=argsx,
                           options={'disp':True,'maxiter':1e3, 'gtol':1e-3})
    # bw = self._set_bw_bounds(bw)  # bound bw if necessary
    return bw.x, bw.success #solution

#@profile(sort='cumulative',filename='./out/experiments.txt')
def profile_run(data,frechets,iterx,name,mode):
    dims=len(data)
    n=len(data[0])
    v_type = f'{"c"*dims}'
    # threshold: number of violations by the cheapest method.
    dens_u_rot = sm.nonparametric.KDEMultivariate(data=data, var_type=v_type, bw='normal_reference')
    cdf_dens_u_rot = dens_u_rot.cdf()
    violations = count_frechet_fails(cdf_dens_u_rot, frechets,methodstring='silverman')

    #profile frechets
    if mode=='exp':
        pr = cProfile.Profile()
        pr.enable()
    bw_frechets,frechet_success = get_bw(data, v_type, dens_u_rot.bw, frech_bounds=frechets)
    if mode == 'exp':
        pr.disable()
    cdf_frechet=sm.nonparametric.KDEMultivariate(data=data, var_type=v_type,
                                                 bw=bw_frechets).cdf()
    violations.update(count_frechet_fails(cdf_frechet, frechets, methodstring='frechet'))

    if mode == 'exp':
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        s = s.getvalue()
        with open(f'./out/{name}_frechet-profile-d{dims}-n{n}-iter{iterx}.txt', 'w+') as f:
            f.write(s)

    #profile cv_ml
    if mode == 'exp':
        pr = cProfile.Profile()
        pr.enable()
    bw_cv_ml,cv_ml_success = get_bw(data, v_type, dens_u_rot.bw)
    if mode == 'exp':
        pr.disable()
    cdf_cv_ml=sm.nonparametric.KDEMultivariate(data=data, var_type=v_type,
                                                 bw=bw_cv_ml).cdf()
    violations.update(count_frechet_fails(cdf_cv_ml, frechets,methodstring='cv_ml'))

    if mode == 'exp':
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        s = s.getvalue()

        with open(f'./out/{name}_loo-ml-profile-d{dims}-n{n}-iter{iterx}.txt', 'w+') as f:
            f.write(s)

    return bw_frechets,bw_cv_ml, violations


def generate_experiments(reps,n,params, distr, dims, name='horns_horns1',mode='exp'):
    bws_frechet={f'bw_{x}':[] for x in params}
    bws_cv_ml={f'bw_{x}':[] for x in params}

    points=np.linspace(0.001,0.999,num=n)

    for iteration in range(reps):
        F = {k: distr.cdf(points,*params[k]) for k in params}
        data = {k: distr.rvs(*params[k], size=n) for k in params}

        # get frechets and thresholds
        frechets = get_frechets(np.array(list(F.values())))
        # profile bw with frechets and bw with cross-validation maximum-likelihood
        bw_frechets, bw_cv_ml, violations=profile_run(np.array(list(data.values())),
                                                      frechets,iteration,name,mode)

        for ix,x in enumerate(params):
            bws_frechet[f'bw_{x}'].append(bw_frechets[ix])
            bws_cv_ml[f'bw_{x}'].append(bw_cv_ml[ix])
        #store data for this experiment
        outable=pd.concat([pd.DataFrame(data),pd.DataFrame(violations)],axis=1)
        outable.to_csv(f'./out/{name}_data_d{dims}-n{n}-iter{iteration}-{reps}.csv',index=False)

    #store bandwidths by each method
    pd.DataFrame(bws_frechet).to_csv(f'./out/{name}_bws_frechet_d{dims}-n{n}-iter{reps}.csv',index=False)
    pd.DataFrame(bws_cv_ml).to_csv(f'./out/{name}_bws_cv_ml_d{dims}-n{n}-iter{reps}.csv',index=False)


def aggregate_experiments():
    #todo: finish
    fl = f'./out/d{dims}-n{n}-reps{reps}.csv'
    #with open(fl, 'w+') as f:
    #    s = io.StringIO()
    #    k = pstats.Stats('./out/experiments.txt', stream=s).sort_stats('cumtime')
    #    k.print_stats()
    #    f.write(s.getvalue())

    k = pd.read_csv(fl,
                    header=3, delim_whitespace=True,
                    usecols={'ncalls', 'tottime', 'percall',
                             'cumtime', 'filename:lineno(function)'}
                    ).rename(columns={'filename:lineno(function)': 'from_func'}).dropna()
    tokeep = np.where([1 if not not re.search('likelihood', x) else 0 for x in k.from_func.values])[0]
    k = k.iloc[tokeep]
    repl=['frechet' if not not re.search('frechet_likelihood', x)
          else 'loo_cv_ml' for x in k.from_func.values]
    k['from_func']=np.array(repl)
    k.to_csv(f'./out/summary_d{dims}-n{n}-reps{reps}.csv',
             index=False)

def main():
    reps = 30

    params = {'horns': (0.5, 0.5),
              'horns1': (0.45, 0.55),
              #'shower': (5., 2.),
              #'grower': (2., 5.),
              #'symetric': (2., 2.)
              }
    dims = len(params)
    ns = [125]#,125, 250, 500, 1000]
    distr = spst.beta
    for n in ns:
        generate_experiments(reps,n,params,distr,dims,
                             name='horns_horns1',mode='debug')

    #lets_be_tidy(rd,frechets)

if __name__ == "__main__":
    main()