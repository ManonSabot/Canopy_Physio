# -*- coding: utf-8 -*-

"""
Run the coupling models between canopy water and carbon fluxes at each
time step given by the forcing file.
Soil hydrology is represented by a simple tipping bucket. The land
surface cover is assumed homogeneous.

This file is part of the TractLSM model.

Copyright (c) 2023 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Run a tractable LSM for a homogeneous surface"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (11.07.2023)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import collections  # ordered dictionaries
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM.SPAC import canopy_intercept  # throughfall
from TractLSM.SPAC import wetness, water_potential, soil_evap  # soil
from TractLSM.SPAC import absorbed_radiation_2_leaves  # canopy rad
from TractLSM.CH2OCoupler import solve_uso  # USO/Medlyn model
from TractLSM.CH2OCoupler import maximise_profit  # ProfitMax/Sperry

try:  # support functions
    from run_utils import time_step, write_csv

except (ImportError, ModuleNotFoundError):
    from TractLSM.run_utils import time_step, write_csv


# ======================================================================

def over_time(df, step, Nsteps, dic, photo, resolution, calc_sw, calc_canT,
              calc_canE, calc_canGs, calc_leafGs):

    """
    Optimisation wrapper at each time step that updates the soil
    moisture and soil water potential for each of the models before
    running them in turn: (i) the Medlyn/USO model (solve_uso), (ii)
    the Profit maximisation (maximise_profit). None of these are run
    for timesteps when PPFD = 0.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all input data & params

    step: int
        current time step

    Nsteps: int
        total number of steps. This is necessary to know whether unit
        conversion must be based on half-hourly time steps or longer
        time steps!

    dic: dictionary
        initially empty upon input, this dictionary allows to return the
        outputs in a trackable manner. From a time-step to another, it
        also keeps in store the soil moisture and transpiration relative
        to each model, in order to accurately update the soil water
        bucket.

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    resolution: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    calc_sw: bool
        if True, computes the soil hydrology, otherwise uses prescribed
        soil moisture / water potential profile

    calc_canT: bool
        if True, computes the canopy temperature, otherwise uses
        prescribed canopy temperature data

    calc_canE: bool
        if True, computes the canopy transpiration, otherwise uses
        prescribed canopy transpiration data

    calc_canGs: bool
        if True, computes the canopy surface conductance, otherwise
        uses prescribed canopy surface conductance data

    calc_leafGs: bool
        if True, computes the leaf stomatal conductance, otherwise
        uses prescribed leaf stomatal conductance data

    Returns:
    --------
    Outputs a tuple of variables depending on the input dic structure.
    When PPFD is zero, a tuple of zero values is returned. If the models
    behave in a non-physical manner, zeros are returned too. Overall,
    the following variables are returned at each time step:

    An(model): float
        net photosynthetic assimilation rate [umol m-2 s-1]

    E(model): float
        transpiration rate [mmol m-2 s-1]

    Ci(model): float
        intercellular CO2 concentration [Pa]

    gs(model): float
        stomatal conductance to water vapour [mol m-2 s-1]

    Pleaf(model): float
        leaf water potential [MPa]

    Tleaf(model): float
        leaf (canopy) temperature [degC]

    Rublim(model): float
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    Eci(model): float
        evaporation rate from the canopy interception of rainfall
        [mmol m-2 s-1]

    Es(model): float
        soil evaporative rate [mmol m-2 s-1]

    sw(model): float
        volumetric soil water content [m3 m-3]

    Ps(model): float
        soil water potential [MPa]

    """

    # parameters & met data
    p = time_step(df, step)

    # tuple of return values
    tpl_return = ()

    # How many timesteps in a day? (for year, month, hour-based dataset)
    try:
        if step >= Nsteps - 1:  # last step of time series
            delta = p.hod - df.iloc[step - 1, df.columns.get_loc('hod')]

        else:
            if df.iloc[step + 1, df.columns.get_loc('hod')] < p.hod:
                delta = df.iloc[step + 1, df.columns.get_loc('hod')]

            else:  # during day
                delta = df.iloc[step + 1, df.columns.get_loc('hod')] - p.hod

        try:
            Dsteps = int(24. / delta)

        except Exception:  # there can be errors for the last step
            Dsteps = 48

    except Exception:  # there can be errors for the last step
        Dsteps = 48

    # canopy interception
    if (p.Tair > 0.) and (p.precip > p.can_sat) and (p.LAI > 0.001):
        throughfall, Eci = canopy_intercept(p)
        p.precip = throughfall  # precip getting thru to soil

    else:
        Eci = 0.

    try:  # is Tsoil one of the input fields?
        Tsoil = p.Tsoil

        if np.isnan(p.Tsoil):
            raise ValueError

    except (IndexError, AttributeError, ValueError):  # estimate Tsoil
        if ((step + 1) % Dsteps == 0) and (step > 0) and (step < Nsteps - 1):

            # average soil temperature assumed ~ average air temperature
            Tsoil = (df.iloc[step - (Dsteps - 1): step + 1,
                             df.columns.get_loc('Tair')].sum() / float(Dsteps))

        else:  # this fails
            Tsoil = p.Tair

    for key in dic.keys():  # loops over the models

        if calc_sw:  # compute soil moisture state
            if step == 0:
                try:
                    dic[key]['sw'] = p.sw0  # antecedent known sw

                except AttributeError:
                    if not np.isclose(p.Ps, p.Psie):  # sw from Ps
                        dic[key]['sw'] = water_potential(p, None)

                    else:
                        dic[key]['sw'] = p.fc  # field capacity

                for layer in ['sw0', 'sw1', 'sw2', 'sw3', 'sw4', 'sw5']:

                    dic[key][layer] = dic[key]['sw']  # ini layers

                dic[key]['Tsoil'] = Tsoil  # same Tsoil for all keys

                # soil albedo?
                if dic[key]['sw0'] < 0.5 * (p.fc - p.pwp):  # 'dry soil'
                    p.albedo_s = p.albedo_ds

                else:  # 'wet soil'
                    p.albedo_s = p.albedo_ws

                __, __, __, __, __, __, __, dic[key]['Es'] = \
                    wetness(p, Dsteps, dic[key]['sw0'], dic[key]['sw1'],
                            dic[key]['sw2'], dic[key]['sw3'], dic[key]['sw4'],
                            dic[key]['sw5'], 0., 0., Tsoil)

            if ((step + 1) % Dsteps != 0) and (step > 0):
                Tsoil = dic[key]['Tsoil']  # keep same Tsoil thru day

            if step > 0:
                dic[key]['Tsoil'] = Tsoil  # same Tsoil for all keys

                # soil albedo?
                if dic[key]['sw0'] < 0.5 * (p.fc - p.pwp):  # 'dry soil'
                    p.albedo_s = p.albedo_ds

                else:  # 'wet soil'
                    p.albedo_s = p.albedo_ws

                dic[key]['sw'], dic[key]['sw0'], dic[key]['sw1'], \
                    dic[key]['sw2'], dic[key]['sw3'], dic[key]['sw4'], \
                    dic[key]['sw5'], dic[key]['Es'] = wetness(p, Dsteps,
                                                              dic[key]['sw0'],
                                                              dic[key]['sw1'],
                                                              dic[key]['sw2'],
                                                              dic[key]['sw3'],
                                                              dic[key]['sw4'],
                                                              dic[key]['sw5'],
                                                              dic[key]['Es'],
                                                              dic[key]['E'],
                                                              Tsoil)

            # soil water pot. corresponding to the key's soil moisture
            p.Ps = water_potential(p, dic[key]['sw'])
            dic[key]['Ps'] = p.Ps

        else:  # the soil moisture profile is prescribed
            dic[key]['sw0'] = p.sw0
            dic[key]['sw'] = p.sw
            dic[key]['Ps'] = p.Ps
            dic[key]['Es'] = soil_evap(p, dic[key]['sw0'])  # mmol m-2 s-1

     # no photosynthesis under these conditions
    if (p.PPFD <= 50.) or (p.VPD <= 0.05) or (p.LAI <= 0.001):

        for key in dic.keys():

            dic[key]['E'], dic[key]['gs'], dic[key]['Pleaf'], dic[key]['A'], \
                dic[key]['Ci'], dic[key]['Rublim'], dic[key]['Tleaf'], \
                dic[key]['Rleaf'] = (0., ) * 8

    else:  # day time

        for key in dic.keys():  # call the model(s)

            # use right Ps
            p.Ps = dic[key]['Ps']

            # soil albedo?
            if dic[key]['sw0'] < 0.5 * (p.fc - p.pwp):
                p.albedo_s = p.albedo_ds

            else:
                p.albedo_s = p.albedo_ws

            try:
                if key == 'uso':  # USO/Medlyn model
                    dic[key]['E'], dic[key]['gs'], dic[key]['Pleaf'], \
                        dic[key]['A'], dic[key]['Ci'], dic[key]['Rublim'], \
                        dic[key]['Tleaf'], dic[key]['Rleaf'] = \
                        solve_uso(p, photo=photo, calc_temp=calc_canT,
                                  calc_trans=calc_canE, calc_Gsurf=calc_canGs,
                                  calc_gs=calc_leafGs)

                if key == 'pmax':  # ProfitMax/Sperry model
                    isun = df.columns.get_loc('fvc_sun')  # ini embolism
                    isha = df.columns.get_loc('fvc_sha')
                    ik = df.columns.get_loc('ksc')
                    p.fvc_sun = df.iloc[np.maximum(0, step-1), isun]
                    p.fvc_sha = df.iloc[np.maximum(0, step-1), isha]

                    fvc, ksc, dic[key]['E'], dic[key]['gs'], \
                        dic[key]['Pleaf'], dic[key]['A'], dic[key]['Ci'], \
                        dic[key]['Rublim'], dic[key]['Tleaf'], \
                        dic[key]['Rleaf'] = \
                            maximise_profit(p, photo=photo, res=resolution,
                                            calc_temp=calc_canT,
                                            calc_trans=calc_canE,
                                            calc_Gsurf=calc_canGs,
                                            calc_gs=calc_leafGs)

                    # keep track of the embolism, ksc
                    df.iloc[step:np.minimum(step+Dsteps, Nsteps), isun] = \
                        fvc[0]
                    df.iloc[step:np.minimum(step+Dsteps, Nsteps), isha] = \
                        fvc[1]
                    df.iloc[step:np.minimum(step+Dsteps, Nsteps), ik] = ksc

            except (TypeError, IndexError, ValueError):  # no solve
                dic[key]['E'], dic[key]['gs'], dic[key]['Pleaf'], \
                    dic[key]['A'], dic[key]['Ci'], dic[key]['Rublim'], \
                    dic[key]['Tleaf'], dic[key]['Rleaf'] = (0., ) * 8

    for key in dic.keys():  # model outputs

        tpl_return += (dic[key]['A'], dic[key]['E'], dic[key]['Ci'],
                       dic[key]['gs'], dic[key]['Pleaf'], dic[key]['Tleaf'],
                       dic[key]['Rublim'], Eci, dic[key]['Es'], dic[key]['sw'],
                       dic[key]['Ps'],)

    return tpl_return


def run(fname, df, Nsteps, models=['USO', 'ProfitMax'], soilwater=None,
        canopyT=None, canopytrans=None, canopygs=None, leafgs=None,
        photo='Farquhar', resolution=None):

    """
    Runs the profit maximisation algorithm within a simplified LSM,
    alongsite the USO model which follows traditional photosynthesis
    and transpiration coupling schemes.

    Arguments:
    ----------
    fname: string
        output filename

    df: pandas dataframe
        dataframe containing all input data & params

    Nsteps: int
        total number of time steps over which the models will be run

    models: list of strings
        names of the models to call

    soilwater: string
        either the dynamic soil hydrology scheme is run, or a
        'prescribed' soil profile is used

    canopyT: string
        either the dynamic canopy energy balance scheme is run, or a
        'prescribed' canopy temperature is used

    canopytrans: string
        either the dynamic canopy transpiration scheme is run, or a
        'prescribed' canopy transpiration is used

    canopygs: string
        either the dynamic stomatal conductance scheme is run, or a
        'prescribed' canopy surface conductance is used

    leafgs: string
        either the dynamic stomatal conductance scheme is run, or a
        'prescribed' leaf stomatal conductance is used

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    resolution: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    Returns:
    --------
    df2: pandas dataframe
        dataframe of the outputs:
            An(model), E(model), Ci(model), gs(model), Pleaf(model),
            Tleaf(model), Rublim(model), ...

    """

    if resolution is None:  # hydraulic stream resolution
        resolution = 'low'

    # initial assumption, no embo
    df['ksc'] = df['kmax'].iloc[0]
    df['fvc_sun'] = 1.
    df['fvc_sha'] = 1.

    # soil albedo will change depending on soil wetness
    df['albedo_s'] = df['albedo_ws'].iloc[0]

    # attributes that won't change in time
    df['soil_volume'] = df['Zbottom'].iloc[0] * df['ground_area'].iloc[0]
    df['soil_top_volume'] = df['Ztop'].iloc[0] * df['ground_area'].iloc[0]

    if soilwater is None:
        calc_sw = True

        try:  # initialise the soil moisture
            df['sw0'] = df['sw'].loc[df['sw'].first_valid_index()]

        except KeyError:
            pass

        try:
            df.drop(['sw', 'Ps'], axis=1, inplace=True)

        except KeyError:
            pass

    else:
        calc_sw = False

        if len(df) - df['Ps'].count() != 0:  # Ps is missing
            df['Ps'] = water_potential(df.iloc[0], df['sw'])

    if canopyT is None:
        calc_canT = True

    else:
        calc_canT = False

    if canopytrans is None:
        calc_canE = True

    else:
        calc_canE = False

    if canopygs is None:
        calc_canGs = True

    else:
        calc_canGs = False

    if leafgs is None:
        calc_leafGs = True

    else:
        calc_leafGs = False

    # non time-sensitive: last valid value propagated until next valid
    df.fillna(method='ffill', inplace=True)

    # two empty dics, to structure the run setup and retrieve the output
    dic = {}  # appropriately run the models
    output_dic = collections.OrderedDict()  # unpack the output in order

    # sub-dic structures
    subdic = {'sw': None, 'sw0': None, 'sw1': None, 'sw2': None, 'sw3': None,
              'sw4': None, 'sw5': None, 'Ps': None, 'Tsoil': None, 'Es': None,
              'E': None, 'gs': None, 'Pleaf': None, 'A': None, 'Ci': None,
              'Rublim': None, 'Tleaf': None, 'Rlref': df['Rlref'].iloc[0]}

    # for the output dic, the order of things matters!
    subdic2 = collections.OrderedDict([('A', None), ('E', None),
                                       ('Ci', None), ('gs', None),
                                       ('Pleaf', None), ('Tleaf', None),
                                       ('Rublim', None), ('Eci', None),
                                       ('Es', None), ('sw', None),
                                       ('Ps', None)])  # output

    # create dictionaries of Nones with the right structures
    if ('USO' in models) or ('USO'.lower() in models):
        dic['uso'] = subdic.copy()
        output_dic['uso'] = subdic2.copy()

    if ('ProfitMax' in models) or ('ProfitMax'.lower() in models):
        dic['pmax'] = subdic.copy()
        output_dic['pmax'] = subdic2.copy()

    # run the model(s) over the range of timesteps / the timeseries
    tpl_out = list(zip(*[over_time(df, step, Nsteps, dic, photo, resolution,
                                   calc_sw, calc_canT, calc_canE, calc_canGs,
                                   calc_leafGs)
                         for step in range(Nsteps)]))

    # unpack the output tuple 17 by 17
    track = 0  # initialize

    for key in output_dic.keys():

        output_dic[key]['A'] = tpl_out[track]
        output_dic[key]['E'] = tpl_out[track + 1]
        output_dic[key]['Ci'] = tpl_out[track + 2]
        output_dic[key]['gs'] = tpl_out[track + 3]
        output_dic[key]['Pleaf'] = tpl_out[track + 4]
        output_dic[key]['Tleaf'] = tpl_out[track + 5]
        output_dic[key]['Rublim'] = tpl_out[track + 6]
        output_dic[key]['Eci'] = tpl_out[track + 7]
        output_dic[key]['Es'] = tpl_out[track + 8]
        output_dic[key]['sw'] = tpl_out[track + 9]
        output_dic[key]['Ps'] = tpl_out[track + 10]
        track += 11

    # save the outputs to a csv file and get the corresponding dataframe
    df2 = write_csv(fname, df, output_dic)

    return df2
