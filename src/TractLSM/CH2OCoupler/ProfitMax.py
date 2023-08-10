# -*- coding: utf-8 -*-

"""
The profit maximisation algorithm (between carbon gain and hydraulic
cost), adapted from Sperry et al. (2017)'s hydraulic-limited stomatal
optimisation model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.

"""

__title__ = "Profit maximisation algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (02.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import sys  # check for version on the system
import numpy as np  # array manipulations, math operators
import bottleneck as bn  # faster C-compiled np for all nan operations

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import f, Weibull_params, hydraulics
from TractLSM.SPAC import absorbed_radiation_2_leaves  # radiation
from TractLSM.SPAC import leaf_energy_balance, leaf_temperature
from TractLSM.SPAC import LH_water_vapour, vpsat, slope_vpsat, psychometric
from TractLSM.SPAC import conductances
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit  # physio
from TractLSM.SPAC.leaf import arrhen


# ======================================================================

def Ci_stream(p, Cs, Tleaf, res, solstep):

    """
    Creates arrays of possible Ci over which to solve the optimization
    criterion.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Cs: float
        CO2 concentration at the leaf surface [Pa]

    Tleaf: float
        leaf temperature [degC]

    res: string
        either 'low' (default), 'med', or 'high' to solve for Ci

    solstep: string
        'var' means the stepping increment between the lower Ci and Cs
        changes depending on Cs (e.g. there are N fixed values between
        the lower Ci and Cs), while 'fixed' means the stepping increment
        between the lower Ci and Cs is fixed (e.g. N values between the
        lower Ci and Cs is not fixed)

    Returns:
    --------
    An array of all potential Ci [Pa] values (Ci values can be anywhere
    between a lower bound and Cs) at a resolution chosen by the user.

    """

    # CO2 compensation point
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC
    gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

    # declare all potential Ci values
    if res == 'low':
        iCi = 0.1
        NCis = 500

    if res == 'med':
        iCi = 0.02
        NCis = 8000

    if res == 'high':
        iCi = 0.001
        NCis = 50000

    if solstep == 'fixed':

        return np.arange(gamstar, Cs, iCi)

    else:

        return np.linspace(gamstar, Cs, NCis, endpoint=False)


def trans_A(p, A, Ci, Dleaf, gs, gb):

    """

    """

    E = (A * conv.FROM_MILI * (gb * conv.GwvGc + gs * conv.GbvGbc) /
         (gs + gb) * Dleaf / (p.CO2 - Ci))

    return E


def hydraulic_cost(p, P):

    """
    Calculates the hydraulic cost function that reflects the increasing
    damage from cavitation and greater difficulty of moving up the
    transpiration stream with decreasing values of the hydraulic
    conductance. Also calculates the associated plant vulnerability.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    P: array
        leaf water potential [MPa], an array of values from the soil
        water potential Ps to the critical water potential Pcrit for
        which cavitation of the xylem occurs

    Returns:
    --------
    cost: array
        hydraulic cost [unitless]

    f(P, b, c): array
        vulnerability curve of the the plant [unitless]

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    # current maximum plant hydraulic conductance (@ saturation)
    kmax = p.kmax * f(p.Ps, b, c)  # mmol s-1 m-2 MPa-1

    # plant vulnerability curve
    VC = f(P, b, c)

    # hydraulic conductance, from kmax @ Ps to kcrit @ Pcrit
    k = p.kmax * VC  # mmol s-1 m-2 MPa-1

    # critical percentage below which cavitation occurs
    kcrit = p.kmax * p.ratiocrit  # xylem cannot recover past this point

    # cost, from kmax @ Ps to kcrit @ Pcrit, normalized, unitless
    cost = (kmax - k) / (kmax - kcrit)

    return cost, VC


def A_trans(p, trans, Ci, Tleaf=None):

    """
    Calculates the assimilation rate given the supply function.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], an array of values depending on
        the possible leaf water potentials (P) and the Weibull
        parameters b, c

    Ci: float
        intercellular CO2 concentration [Pa], corresponding to a
        leaf water potential (P) for which the transpiration cost is
        minimal and the C assimilation gain is maximal

    Tleaf: float
        leaf temperature [degC]

    Returns:
    --------
    Calculates the photosynthetic gain A [umol m-2 s-1] for a given
    Ci(P) over an array of trans(P) values.

    """

    # get CO2 diffusive conduct.
    gc, __, __ = leaf_energy_balance(p, trans, Tleaf=Tleaf)
    A_P = conv.U * gc * (p.CO2 - Ci) / (p.Patm * conv.MILI)

    try:
        A_P[np.isclose(np.squeeze(gc), cst.zero, rtol=cst.zero,
            atol=cst.zero)] = cst.zero

    except TypeError:
        pass

    return A_P


def mtx_minimize(p, trans, all_Cis, photo):

    """
    Uses matrices to find each value of Ci for which An(supply) ~
    An(demand) on the transpiration stream.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    all_Cis: array
        all potential Ci values over the transpiration stream (for each
        water potential, Ci values can be anywhere between a lower bound
        and Cs)

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Returns:
    --------
    The value of Ci for which An(supply) is the closest to An(demand)
    (e.g. An(supply) - An(demand) closest to zero).

    """

    demand, __, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Cis, photo)
    supply = A_trans(p, np.expand_dims(trans, axis=1), all_Cis)

    # closest match to ~ 0. (i.e. supply ~ demand)
    idx = bn.nanargmin(abs(supply - demand), axis=1)

    # each Ci on the transpiration stream
    Ci = np.asarray([all_Cis[e, idx[e]] for e in range(len(trans))])
    Ci = np.ma.masked_where(idx == 0, Ci)

    return Ci


def split(a, N):

    """
    Splits a list or array into N-roughly equal parts.

    Arguments:
    ----------
    a: list or array
        list/array to be split, can contain any data type

    N: int
        number of sub-lists/arrays the input must be split into

    Returns:
    --------
    A list of N-roughly equal parts.

    """

    integ = int(len(a) / N)
    remain = int(len(a) % N)

    splitted = [a[i * integ + min(i, remain):(i + 1) * integ +
                  min(i + 1, remain)] for i in range(N)]

    return splitted


def photo_gain(p, trans, photo, res, solstep):

    """
    Calculates the photosynthetic C gain of a plant, where the
    photosynthetic rate (A) is evaluated over the array of leaf water
    potentials (P) and, thus transpiration (E), and normalized by the
    instantaneous maximum A over the full range of E.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    solstep: string
        'var' means the stepping increment between the lower Ci and Cs
        changes depending on Cs (e.g. there are N fixed values between
        the lower Ci and Cs), while 'fixed' means the stepping increment
        between the lower Ci and Cs is fixed (e.g. N values between the
        lower Ci and Cs is not fixed)

    Returns:
    --------
    gain: array
        unitless instantaneous photosynthetic gains for possible values
        of Ci minimized over the array of E

    Ci: array
        intercellular CO2 concentration [Pa] for which A(P) is minimized
        to be as close as possible to the A predicted by either the
        Collatz or the Farquhar photosynthetis model

    """

    # accounting for canopy and leaf conductances is needed further
    __, gs, gb = leaf_energy_balance(p, trans)  # mol m-2 s-1

    # ref. photosynthesis for which the dark respiration is set to 0
    A_ref, __, __, __ = calc_photosynthesis(p, trans, p.CO2, photo, Rleaf=0.)

    # Cs < Ca
    boundary_CO2 = p.Patm * conv.FROM_MILI * A_ref / (gb * conv.GbcvGb)
    Cs = np.minimum(p.CO2, p.CO2 - boundary_CO2)  # Pa

    # potential Ci values over the full range of transpirations
    if res == 'low':
        iCi = 0.1
        NCis = 500

    if res == 'med':
        iCi = 0.02
        NCis = 8000

    if res == 'high':
        iCi = 0.001
        NCis = 50000

    if solstep == 'fixed':
        Cis = np.tile(np.arange(0.1, bn.nanmax(Cs) + iCi, iCi),
                      len(trans)).reshape(len(trans), -1)
        Cis = np.ma.masked_where(Cis > np.repeat(Cs, Cis.shape[1])
                                         .reshape(len(Cs), -1) + iCi, Cis)
        Cis[np.arange(len(trans)), bn.nanargmax(Cis, axis=1)] = Cs

    else:
        Cis = np.asarray([np.linspace(0.1, Cs[e], NCis) for e in
                          range(len(trans))])


    Ci = mtx_minimize(p, trans, Cis, photo)
    Ci = np.ma.masked_where(np.logical_or(Ci >= Cs, np.isnan(Ci)), Ci)

    try:
        A_P = A_trans(p, trans, Ci)  # get A demand (A(P))

        # update mask
        Ci[1:] = np.ma.masked_where(np.isclose(A_P[1:], 0.), Ci[1:])
        A_P[1:] = np.ma.masked_where(np.isclose(A_P[1:], 0.), A_P[1:])

        gain = A_P / np.ma.amax(A_P[1:])  # photo gain, soil P excluded

        if np.ma.amax(A_P[1:]) < 0.:  # when resp >> An everywhere
            gain *= -1.

    except ValueError:  # if trans is "pre-opimised" for
        gain = 0.

    return gain, Ci


def trans_Ebal(p, Tleaf, gs):

    """
    Calculates transpiration following Penman-Monteith at the leaf level
    accounting for effects of leaf temperature and feedback on
    evaporation.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    Tleaf: float
        leaf temperature [degC]

    gs: float
        stomatal conductance [mol m-2 s-1]

    Returns:
    --------
    trans: float
        transpiration rate [mol m-2 s-1]

    real_zero: boolean
        True if the transpiration is really zero, False if Rnet is
        negative

    gw: float
        total leaf conductance to water vapour [mol m-2 s-1]

    gb: float
        boundary layer conductance to water vapour [mol m-2 s-1]

    """

    # check that the trans value satisfies the energy balance
    real_zero = True

    # get conductances
    gw, gH, gb, __ = conductances(p, Tleaf=Tleaf, gs=gs)  # mol m-2 s-1

    # latent heat of water vapor
    Lambda = LH_water_vapour(p)  # J mol-1

    # slope of saturation vapour pressure of water vs Tair
    slp = slope_vpsat(p)  # kPa degK-1

    if np.isclose(gs, 0., rtol=cst.zero, atol=cst.zero):
        trans = cst.zero

    else:
        gamm = psychometric(p)  # psychrometric constant, kPa degK-1
        trans = (slp * p.Rnet + p.VPD * gH * cst.Cp) / (Lambda *
                                                        (slp + gamm * gH / gw))

        if trans < 0.:  # Penman-Monteith failed, non-physical trans
            real_zero = False

        trans = max(cst.zero, trans)  # mol m-2 s-1

    return trans, real_zero, gw, gb


def maximise_profit(p, photo='Farquhar', res='low', solstep='var',
                    threshold_conv=0.015, iter_max=40, scaleup=True,
                    calc_temp=True, calc_trans=True, calc_Gsurf=True,
                    calc_gs=True):

    """
    Finds the instateneous profit maximisation, following the
    optimisation criterion for which, at each instant in time, the
    stomata regulate canopy gas exchange and pressure to achieve the
    maximum profit, which is the maximum difference between the
    normalised photosynthetic gain (gain) and the hydraulic cost
    function (cost). That is when d(gain)/dP = d(cost)/dP.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    solstep: string
        'var' means the stepping increment between the lower Ci and Cs
        changes depending on Cs (e.g. there are N fixed values between
        the lower Ci and Cs), while 'fixed' means the stepping increment
        between the lower Ci and Cs is fixed (e.g. N values between the
        lower Ci and Cs is not fixed)

    threshold_conv: float
        convergence threshold for the new leaf temperature to be in
        energy balance

    iter_max: int
        maximum number of iterations allowed on the leaf temperature
        before reaching the conclusion that the system is not energy
        balanced

    scaleup: boolean
        True yields canopy-scale variables whilst False will lead the
        same variables but at the tree-scale

    calc_temp: boolean
        if True, computes the canopy temperature, otherwise uses
        prescribed canopy temperature data

    calc_trans: boolean
        if True, computes the canopy transpiration, otherwise uses
        prescribed canopy transpiration data

    calc_Gsurf: boolean
        if True, computes the canopy surface conductance, otherwise
        uses prescribed canopy surface conductance data

    calc_gs: boolean
        if True, computes the leaf stomatal conductance, otherwise
        uses prescribed leaf stomatal conductance data

    Returns:
    --------
    fvuln: array
        reference vulnerability point for the optimisation for each leaf

    ksc: float
        instantaneous maximum root-to-leaf hydraulic conductance
        [mmol m-2 s-1 MPa-1] at maximum profit across leaves

    E_can: float
        transpiration [mmol m-2 s-1] at maximum profit across leaves

    gs_can: float
        stomatal conductance [mol m-2 s-1] at maximum profit across
        leaves

    Pleaf_can: float
        leaf water potential [MPa] at maximum profit across leaves

    An_can: float
        net photosynthetic assimilation rate [umol m-2 s-1] at maximum
        profit across leaves

    Ci_can: float
        intercellular CO2 concentration [Pa] at maximum profit across
        leaves

    rublim_can: string
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    Tleaf_can: float
        leaf temperature [degC] at maximum profit across leaves

    Rleaf_can: float
        day respiration rate [umol m-2 s-1] at maximum profit across
        leaves

    """

    success = True  # initial assumption: the optimisation will succeed

    # retrieve relevant sunlit / shaded fractions
    fRcan, fPPFD, fLAI, fscale2can, fgradis = absorbed_radiation_2_leaves(p)

    # saturation vapour pressure of water at Tair
    esat_a = vpsat(p.Tair)  # kPa

    # hydraulics
    P, trans = hydraulics(p, res=res)
    COST, VC = hydraulic_cost(p, P)

    # sunlit / shaded outputs
    fvc = np.zeros(len(fPPFD))
    E = np.zeros(len(fPPFD))
    gs_can = np.zeros(len(fPPFD))
    gs_can[:] = np.nan  # make sure we have nans for averaging
    Pleaf = np.zeros(len(fPPFD))
    Pleaf[:] = np.nan  # make sure we have nans for averaging
    An = np.zeros(len(fPPFD))
    Aj = np.zeros(len(fPPFD))
    Ac = np.zeros(len(fPPFD))
    Ci_can = np.zeros(len(fPPFD))
    Ci_can[:] = np.nan  # make sure we have nans for averaging
    Tleaf = np.zeros(len(fPPFD))
    Tleaf[:] = np.nan  # make sure we have nans for averaging
    Rleaf = np.zeros(len(fPPFD))

    # original LAI, PPFD
    LAI = p.LAI
    PPFD = p.PPFD

    if calc_trans:  # scaling to match A (conductance of can, not leaf)
        trans_can = [trans * e for e in fscale2can]

    else:
        trans_can = [conv.FROM_MILI * float(p.E) * np.ones_like(trans) *
                     e for e in fscale2can]
        idx = np.nanargmin(np.abs(trans - conv.FROM_MILI * float(p.E)))
        trans_can[0][:idx] = np.nan
        trans_can[1][:idx] = np.nan
        trans_can[0][idx + 1:] = np.nan
        trans_can[1][idx + 1:] = np.nan

    # sunlit / shaded loop, two assimilation streams
    for i in range(len(fRcan)):

        if i == 0:  # sunlit
            fvc_opt = p.fvc_sun

        else:  # shaded
            fvc_opt = p.fvc_sha

        p.Rnet = fRcan[i]
        p.PPFD = fPPFD[i]
        p.LAI = fLAI[i]
        p.scale2can = fscale2can[i]
        p.gradis = fgradis[i]
        trans = trans_can[i]
        cost = np.ma.copy(COST)
        vc = np.ma.copy(VC)

        if p.PPFD > 50.:  # min threshold for photosynthesis
            Cs = p.CO2  # Pa

            if not calc_temp:
                Tleaf[i] = p.Tleaf  # deg C

            elif not calc_trans:
                Tleaf[i], __ = leaf_temperature(p, conv.FROM_MILI * float(p.E))

            else:
                Tleaf[i] = p.Tair  # deg C

            # leaf-air vpd, kPa, gs model not valid ~0.05
            if calc_temp and calc_trans:
                Dleaf = np.maximum(0.05, p.VPD)

            else:
                esat_l = vpsat(Tleaf[i])  # vpsat at Tleaf, kPa
                Dleaf = (esat_l - (esat_a - np.maximum(0.05, p.VPD)))

            if not calc_Gsurf:  # gb? gs?
                try:
                    gb = p.gb

                except AttributeError:
                    __, gb = leaf_temperature(p, 0., Tleaf=Tleaf[i])

                gs_can[i] = gb * p.Gs / (gb - p.Gs)

            elif not calc_gs:  # gb?
                gs_can[i] = p.gs
                __, gb = leaf_temperature(p, 0., gs=gs_can[i], Tleaf=Tleaf[i])

            else:  # gb?
                gs_can[i] = 0.
                __, gb = leaf_temperature(p, 0., Tleaf=Tleaf[i])

            # iter on the solution until it is stable enough
            iter = 0

            while True:

                # optimization model, on a stream of possible Ci values
                Cis = Ci_stream(p, Cs, Tleaf[i], res, solstep)

                # rate of photosynthesis, μmol m-2 s-1
                if calc_gs and calc_Gsurf:
                    A, __, __, __ = calc_photosynthesis(p, 0., Cis, photo,
                                                        Tleaf=Tleaf[i])

                else:
                    A, __, __, __ = \
                        calc_photosynthesis(p, 0., Cis, photo, Tleaf=Tleaf[i],
                                            gsc=conv.U * conv.GcvGw *
                                                gs_can[i])

                # photo gain
                gains = A / np.ma.amax(A)

                # when resp >> An everywhere
                if np.ma.amax(A) < 0.:
                    gains *= -1.

                # trans, hydraulic cost, and associated functions
                Ecan = trans_A(p, A, Cis, Dleaf, gs_can[i], gb)  # mol m-2 s-1
                Ecan[Ecan < 0.] = cst.zero
                idxs = [np.nanargmin(np.abs(e - trans)) for e in Ecan]
                costs = np.array([cost[idxs]][0])
                LWPs = np.array([P[idxs]][0])
                transcan = np.array([trans[idxs]][0])
                vcs = np.array([vc[idxs]][0])

                # look for the most net profit
                profit = gains - costs

                # deal with edge cases by rebounding the solution
                gc, gs, gb = leaf_energy_balance(p, transcan)

                if calc_gs and calc_Gsurf:
                    mask = np.logical_and(np.logical_and(Ecan[1:] >= cst.zero,
                                          LWPs[1:] >= P[-1]),
                                          np.logical_and(gc[1:] >= cst.zero,
                                          Cis[1:] / p.CO2 < 0.95))

                else:
                    mask = np.logical_and(np.logical_and(Ecan[1:] >= cst.zero,
                                          LWPs[1:] >= P[-1]),
                                          Cis[1:] / p.CO2 < 0.95)

                profit_check = profit[1:][mask]

                try:
                    iopt = np.isclose(profit, max(profit_check))
                    iopt = [list(iopt).index(e) for e in iopt if e]

                    if iopt:  # opt values
                        fvc[i] = vcs[iopt[0]]
                        Ci_can[i] = Cis[iopt[0]]
                        Pleaf[i] = LWPs[iopt[0]]

                        if calc_gs and calc_Gsurf:
                            gs_can[i] = gs[iopt[0]]

                        # calculate new trans, gw, gb, Tleaf
                        E[i], real_zero, gw, gb = trans_Ebal(p, Tleaf[i],
                                                             gs_can[i])

                        if not calc_trans:
                            E[i] = conv.FROM_MILI * float(p.E)

                        new_Tleaf, gb = leaf_temperature(p, E[i], gs=gs_can[i],
                                                         Tleaf=Tleaf[i],
                                                         gradis=True)

                        # rubisco- or electron transport-limitation?
                        An[i], Aj[i], Ac[i], Rleaf[i] = \
                                calc_photosynthesis(p, E[i], Ci_can[i], photo,
                                                    Tleaf=Tleaf[i])

                        # new Cs (in Pa)
                        boundary_CO2 = (p.Patm * conv.FROM_MILI * An[i] /
                                            (gb * conv.GbcvGb))
                        Cs = np.maximum(cst.zero, np.minimum(p.CO2,
                                            p.CO2 - boundary_CO2))

                        # new Dleaf
                        if calc_temp and calc_trans and (np.isclose(trans[i],
                                cst.zero, rtol=cst.zero, atol=cst.zero) or
                                np.isclose(gw, cst.zero, rtol=cst.zero,
                                atol=cst.zero) or np.isclose(gs[i], cst.zero,
                                rtol=cst.zero, atol=cst.zero)):
                            Dleaf = np.maximum(0.05, p.VPD)  # kPa

                        else:
                            esat_l = vpsat(new_Tleaf)
                            Dleaf = (esat_l - (esat_a - np.maximum(0.05,
                                         p.VPD)))

                        # force stop when E < 0. (non-physical)
                        if (iter < 1) and (not real_zero):
                            real_zero = None

                        # check for convergence
                        if ((real_zero is None) or (iter > iter_max) or
                                ((real_zero) and (abs(Tleaf[i] - new_Tleaf) <=
                                threshold_conv) and not np.isclose(gs[i],
                                cst.zero, rtol=cst.zero, atol=cst.zero))):
                            break

                        if calc_temp and calc_trans:
                            Tleaf[i] = new_Tleaf  # no convergence, iterate

                        iter += 1

                    else:
                        raise ValueError()

                except (ValueError, TypeError):  # no opt
                    if iter < 1 or real_zero:  # no prev. valid outcomes
                        success = False

                    break

            if not success:  # this is rare, use last opt vc
                idx = bn.nanargmin(abs(vc - fvc_opt))
                fvc[i] = vc[idx]
                __, gs_can[i], __ = leaf_energy_balance(p, trans[idx])
                __, Ci_can[i] = photo_gain(p, np.asarray([trans[idx]]), photo,
                                           res, False, solstep)
                Ci_can[i] = Ci_can[0]  # a single value is returned

                if Ci_can[i] >= 0.95:  # no solve
                    Ci_can[i] = np.nan

                if (str(Ci_can[i]) == '--'):  # no valid Ci
                    if not calc_temp:
                        Tleaf[i] = p.Tleaf  # deg C

                    elif not calc_trans:
                        Tleaf[i], __ = leaf_temperature(p,
                                                        conv.FROM_MILI * float(p.E))

                    else:
                        Tleaf[i] = p.Tair

                    fvc[i], gs_can[i], E[i], An[i], Ci_can[i], Rleaf[i] = \
                        (fvc_opt, ) + (0., ) * 5

                Pleaf[i] = P[idx]

                if not calc_temp:
                    Tleaf[i] = p.Tleaf  # deg C

                elif not calc_trans:
                    E[i] = conv.FROM_MILI * float(p.E)
                    Tleaf[i], __ = leaf_temperature(p, E[i])

                else:
                    # recalc. Tleaf, E accounting for ALL feedbacks
                    Tleaf[i], __ = leaf_temperature(p, trans[idx],
                                                    gs=gs_can[i])
                    Tleaf[i], __ = leaf_temperature(p, trans[idx],
                                                    gs=gs_can[i],
                                                    Tleaf=Tleaf[i],
                                                    gradis=True)

                if calc_trans:
                    E[i], __, __, __ = trans_Ebal(p, Tleaf[i], gs_can[i])

                # rubisco- or electron transport-limitation?
                An[i], Aj[i], Ac[i], Rleaf[i] = \
                    calc_photosynthesis(p, E[i], Ci_can[i], photo,
                                        Tleaf=Tleaf[i])

            # if the critical point has been reached, stall
            if np.isclose(fvc[i], p.ratiocrit):
                if calc_temp:
                    Tleaf[i] = p.Tair

                gs_can[i], E[i], An[i], Ci_can[i], Rleaf[i] = (0., ) * 5

            if calc_trans and calc_Gsurf and calc_gs and np.isclose(E[i], 0.):
                if calc_temp:
                    Tleaf[i] = p.Tair

                gs_can[i], E[i], An[i], Ci_can[i], Rleaf[i] = (0., ) * 5

        else:
            if calc_temp:
                Tleaf[i] = p.Tair

            fvc[i], gs_can[i], E[i], An[i], Ci_can[i] = ((fvc_opt, ) +
                                                         (0., ) * 4)

    # deal with no solves for Pleaf
    Pleaf[Pleaf > p.Ps] = p.Ps

    if np.isclose(np.nanmean(Pleaf), p.Ps):
        E_can, gs_can, Pleaf_can, An_can, Ci_can, Tleaf_can, Rleaf_can = \
            (0., ) * 7
        rublim_can = -9999.

    else:  # merge contributions from sunlit and shaded leaves
        with np.errstate(invalid='ignore'):  # nans, no warning

            # set intensive quantities to nan if necessary
            gs_can[np.isclose(gs_can, 0.)] = np.nan
            Pleaf[np.isclose(Pleaf, p.Ps)] = np.nan
            Ci_can[np.isclose(Ci_can, 0.)] = np.nan
            Tleaf[np.isclose(Tleaf, 0.)] = np.nan

            # total contributions
            E_can = np.nansum(E) * conv.MILI  # mmol m-2 s-1
            gs_can = np.nanmean(gs_can)  # mol m-2 s-1
            Pleaf_can = np.nanmean(Pleaf)  # MPa
            An_can = np.nansum(An)  # umol m-2 s-1
            Ci_can = np.nanmean(Ci_can)  # Pa
            rublim_can = rubisco_limit(np.nansum(Aj), np.nansum(Ac))
            Tleaf_can = np.nanmean(Tleaf)  # degC
            Rleaf_can = np.nansum(Rleaf)  # umol m-2 s-1

    if not scaleup:  # downscale fluxes to the tree
        E_can /= np.sum(fscale2can)
        An_can /= np.sum(fscale2can)

    # reset original all canopy / forcing LAI, PPFD
    p.LAI = LAI
    p.PPFD = PPFD

    if (any(np.isnan([E_can, gs_can, An_can, Ci_can])) or
       any([E_can < 0., gs_can < 0., Ci_can < 0.])):
        E_can, gs_can, Pleaf_can, An_can, Ci_can, Tleaf_can, Rleaf_can = \
            (0., ) * 7

    if np.isnan(Pleaf_can):
        Pleaf_can = 0.

    return (fvc, p.kmax * VC[0], E_can, gs_can, Pleaf_can, An_can, Ci_can,
            rublim_can, Tleaf_can, Rleaf_can)
