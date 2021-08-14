# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:20:34 2020

@author: M77100
"""
import numpy as np
import pandas as pd

import matplotlib.dates as mdates

from scipy import stats
from scipy.optimize import curve_fit


class Tendance:
    """ Provides methods to determine increase/decrease tendency in citations for the time series
    """
    def __init__(self):
        print("Tendance class initialised.")

    def compute_tendance(self, df_c, statTable, freq, simpFit=True):
        """ Computes the increase/decrease tendency of a time series

         Parameters
         ----------
         df_c : dataframe
            stacked dataframes of all time series (binned data).

         statTable: dataframe
            statistical indicators of tome series.

        freq : str
            The bin size in weeks or a month {"W", "2W", "3W", "M"}
            default is month ("M")

        simpFit : boolean
            True by default apply a simple linear fit to each time series.
            False : shot noises included in linear fit

        Return
        ----------
        Tendecy dataframe
        """
        self._freq = freq
        print("Tendance : compute_tendance : computing the increase/decrease tendency of the organisations")
        fitFun = self.compute_simple_lin_fit if simpFit else self.compute_lin_fit
        df_c_fit = fitFun(df_c)
        print("    Merging the tendencies with the statTable ... ")
        return statTable.merge(df_c_fit, on="organName") \
            .sort_values("mean", ascending=False)


    def compute_simple_lin_fit(self, df_c):
        """ Fits a line to each time serie

         Parameters
         ----------
         df_c : dataframe
            stacked dataframes of all time series (binned data).

        Return
        ----------
        Dataframe including the parameters and the characteristics of the fitted lines
        """
        print("Tendance : compute_simple_lin_fit : Linear fit to all data points per organisation.")
        print("    Uncertainities on bin counts are excluded.")
        self._df_c = df_c
        tbl_fit = self.prep_to_fit()
        tbl_fit["fitRes"] = tbl_fit["x,y,err"].apply(self.simple_line_fit)
        tbl_fit[['slope', "intercept", 'R2', "linePValue", "slopeErr"]] = \
            pd.DataFrame(tbl_fit.fitRes.tolist())
        tbl_fit = tbl_fit.drop(columns=["fitRes", "x,y,err"])

        return tbl_fit

    def compute_lin_fit(self, df_c):
        """ Fits a line to each time serie by including the shot noises for each binCount

         Parameters
         ----------
         df_c : dataframe
            stacked dataframes of all time series (binned data).

        Return
        ----------
        Dataframe including the parameters and the characteristics of the fitted lines
        """
        print("Tendance : compute_lin_fit : Linear fit to all data points per organisation.")
        print("     Shot noises included.")
        self._df_c = df_c
        tbl_fit = self.prep_to_fit()
        tbl_fit["fitRes"] = tbl_fit["x,y,err"].apply(self.line_fit)
        tbl_fit[['slope', "intercept", "slopeErr", "xi2"]] = \
            pd.DataFrame(tbl_fit.fitRes.tolist())
        tbl_fit = tbl_fit.drop(columns=["fitRes", "x,y,err"])

        return tbl_fit

    def prep_to_fit(self):
        """ Prepare the dataframe of time series to fit a line
        Return
        ----------
        Dataframe with 2 columns : organName and a list of date,binCount,uncertainty per organName
        """
        df_befitted = self._df_c.copy()
        # Converting date formats to digits
        df_befitted["modifDate"] = df_befitted["date"] \
            .apply(lambda x: mdates.date2num(x))
        if self._freq == "M" :
            df_befitted["modifDate"] /= 30.
        elif self._freq == "3W" :
            df_befitted["modifDate"] /= 21.
        elif self._freq == "2W" :
            df_befitted["modifDate"] /= 14.
        elif self._freq == "W" :
            df_befitted["modifDate"] /= 7.

        # Assigning Shot noise to each count
        df_befitted["countError"] = df_befitted["binCount"] \
            .apply(lambda x: round(np.sqrt(x), 0))
        # Put all required fit data points to a list
        df_befitted["x,y,err"] = df_befitted[["modifDate", "binCount", "countError"]] \
            .values \
            .tolist()
        # Combining all data points for a given organisation
        df_befitted = df_befitted.groupby("organName")["x,y,err"] \
            .apply(list) \
            .reset_index()

        return df_befitted

    def simple_line_fit(self, liste):
        """ Fits a line to data points through scipy.stat.linregress method (uncertainties excluded).

        Parameters
        ----------
         liste : list of data points (date,binCount,uncertainty) per organName

        Return
        ----------
         List of output slopes, associated errors are in degrees and other fit characteristics
        """

        try:
            x = list(map(lambda f: f[0], liste))
            y = list(map(lambda f: f[1], liste))
            # err = list( map(lambda f : f[2], liste) )
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return [round(np.arctan(slope) * 180 / np.pi, 1), round(intercept, 1),
                    round(r_value * r_value, 2), round(p_value, 2),
                    round(np.arctan(std_err) * 180 / np.pi, 1)]

        except IndexError:
            print("line_fit error : list out of index")
            return np.nan

    def line_fit(self, liste):
        """ Fits a line to data points through scipy.curve_fit method method (uncertainties inluded).

        Parameters
        ----------
         liste : list of data points (date,binCount,uncertainty) per organName

        Return
        ----------
         List of output slopes, associated errors are in degrees and other fit characteristics
        """

        def linear_model(x, a, b):
            return a * x + b

        try:
            x = np.array(list(map(lambda f: f[0], liste)))
            y = np.array(list(map(lambda f: f[1], liste)))
            err = np.array(list(map(lambda f: f[2], liste)))

            fitParam, covParam = curve_fit(linear_model, x, y, sigma=err)
            slope, intercept = fitParam
            slopeErr = np.sqrt(covParam[0][0])
            # interceptErr = covParam[1][1]

            yFit = linear_model(x, slope, intercept)
            d = yFit - y
            xi2Arg = d / err
            xi2 = sum(xi2Arg * xi2Arg) / (len(d) - 2)

            return [round(np.arctan(slope) * 180 / np.pi, 1), round(intercept, 1),
                    round(np.arctan(slopeErr) * 180 / np.pi, 1), xi2]

        except IndexError:
            print("line_fit warning : list out of index")
            return np.nan

        except ZeroDivisionError:
            print("line_fit warning : uncertainties include zero(s) ")
            return np.nan


##############################################################################

def constrained_tendance(df_tendance, noiseRatio=.5, minNumBins=6):
    """ Extract organisations with most clear increase/decrease tendency.
        This method filter those measurement with slopeErr/slope < noiseRatio

    Parameters
    ----------
     df_tendance : dataframe
        tendency dataframe
     noiseRatio : float
        should be in [0,1]
     minNumBins : int
        minimum accepted number of bins (date) for each organisation

    Return
    ----------
     dataframe of constrained tendencies.
    """

    print("constrained_tendance : ")
    print("    ratio between slopErr and slope :", noiseRatio)
    print("    Minimum accepted number of bins:", minNumBins)
    if (noiseRatio > 1.) | (noiseRatio < 0.):
        print(" Error : ratio between slopErr and slope should be 0<r<1")
        return pd.DataFrame(columns=df_tendance.columns)
    df_tendance = df_tendance[
        (df_tendance["slopeErr"] < noiseRatio * abs(df_tendance["slope"])) &
        (df_tendance["numBins"] >= int(minNumBins))
        ] \
        .sort_values(["mean"], ascending=False)
    return df_tendance

##############################################################################
