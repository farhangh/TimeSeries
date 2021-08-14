# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:13:33 2020

@author: M77100
"""

import numpy as np
import pandas as pd


def compute_interpol(df, **gt):
    """ For a given date it computes a binCount by interpolating the binCounts of the previous and next dates
        This will be needed by the internal sigma computation.

    Parameters
    ----------
     groupby key and target columns

    Return
    ----------
     Dataframe including the interpolated binCounts
    """

    print("---> Calling compute_interpol : interpolating between upper and lower adjacents of a row.")

    groupKey = gt["groupKey"]
    targetCol = gt["targetCol"]
    print("    Groupby key :", groupKey, ", target column : ", targetCol)
    df = df[[groupKey, "date", targetCol]]

    df_inpo = df.copy()
    print("    Computing the time difference between next and two next rows ...")
    df_inpo["diff_date"] = df.groupby([groupKey])["date"].diff()
    df_inpo["diff2_date"] = df_inpo.groupby([groupKey])["date"].diff(2)
    df_inpo["diff_date"] = df_inpo["diff_date"].dt.days
    df_inpo["diff2_date"] = df_inpo["diff2_date"].dt.days

    print("    Computing the difference of", targetCol, "between two-next rows ...")
    df_inpo["diff2_" + targetCol] = df_inpo.groupby([groupKey])[targetCol].diff(2)

    print("    Computing the slop of the line connecting the upper neighbour to the lower ...")
    df_inpo["slop"] = (df_inpo["diff2_" + targetCol] / df_inpo["diff2_date"]).replace(np.inf, 0)

    print("    Multiplying the slop with the time difference to the previous row ...")
    df_inpo["increment"] = df_inpo["slop"].shift(-1) * df_inpo["diff_date"]

    print("    Computing the interpolated value ... ")
    df_inpo["intpol_" + targetCol] = df_inpo[targetCol].shift(1) + df_inpo["increment"]

    print("    Returning the interpolated dataframe ...")
    return df_inpo[[groupKey, "date", targetCol, "intpol_" + targetCol]].dropna()


##############################################################################

class StatIndic:
    def __dir__(self):
        print("StatIndic class initiated.")

    def compute_stat(self, df, intp=True, numBins="numBins", **gt):
        """ Computes statistical indicators for each tine serie.

        Parameters
        ----------
        df : dataframe
            dataframe including the time series.
        intp : boolean
            True by default to apply interpolation.
            If False it computes the binCount difference between consecutive dates.
        numBins : str
            Name of the column including total number of measurements (bin) per organisation.
        gt : keywords
            groupby key and target columns


        Return
        ----------
        Dataframe including the mean, sigma, median, internal sigma and etc per organisation.
        """

        print("---> Calling compute_stat : variability search through internal dispersion method.")
        if (intp):
            print("Interpolation : active ")
        else:
            print("Interpolation : inactive ")

        groupKey = gt["groupKey"]
        targetCol = gt["targetCol"]

        print("Groupby key :", groupKey, ", target column : ", targetCol)
        df_org = df.copy()
        df = df[[groupKey, "date", targetCol]]

        print("Computing the {} mean and median values ...".format(targetCol))
        df_mean = df.groupby([groupKey]).agg(["mean", "median", "min", "max"])
        df_mean.columns = ["mean", "median", "min", "max"]
        df_mean.reset_index(inplace=True)
        df_mean["mean"] = df_mean["mean"].apply(lambda x: round(x, 0))

        print("Computing the dispersion ...")
        df_sigma = df.groupby([groupKey]).std()
        df_sigma.columns = ["sigma"]

        df_internal = pd.DataFrame()
        if (intp):
            print("Computing the interpolations ... ")
            df_intpo = compute_interpol(df, groupKey=groupKey, targetCol=targetCol)

            # Internal sigma method to serach for variabilities :
            #  Habibi et al., Astronomy and Astrophysics 525 (2011) A108, equation (5)
            print("Computing the internal dispersion ...")
            df_intpo["diff2"] = (df_intpo[targetCol] - df_intpo["intpol_" + targetCol]).pow(2)
            df_intpo = df_intpo.drop(columns=[targetCol, "intpol_" + targetCol])

            df_internal = df_intpo.groupby([groupKey]) \
                .mean() \
                .pow(.5)
            df_internal.columns = (["internalSigma"])
            #df_internal["internalSigma"] = df_internal["internalSigma"].apply(lambda x: round(x, 2))

        else:
            print("Computing the difference between consecutive", targetCol, " for each", groupKey, "...")
            df_diff2 = df.copy()
            df_diff2["diff2"] = df.groupby([groupKey])[targetCol].diff().pow(2)
            df_diff2 = df_diff2.dropna()

            print("Computing the internal dispersion ...")
            df_internal = df_diff2.groupby([groupKey]).mean().pow(.5).drop(columns=[targetCol])
            df_internal.columns = (["internalSigma"])
            df_internal["internalSigma"] = df_internal["internalSigma"].apply( lambda x : round(x, 0) )


        print("Computing the ratio between the dispersion and the internal dipesion ...")
        df_intsig = df_internal.merge(df_sigma, on=groupKey)
        df_intsig["sigmasRatio"] = 0.
        df_intsig.loc[abs(df_intsig["sigma"])>1.e-6, "sigmasRatio"] =  \
            (df_intsig["sigma"] / df_intsig["internalSigma"]).replace(np.inf, 0)
        df_intsig = df_intsig.dropna().reset_index()
        df_intsig["sigma"] = df_intsig["sigma"].apply(lambda x: round(x, 0))
        df_intsig["sigmasRatio"] = df_intsig["sigmasRatio"].apply(lambda x: round(x, 1))
        df_intsig.drop(columns=["internalSigma"], inplace=True)

        print("Constructing the stat table ...")
        df_stat = df_mean.merge(df_intsig, on=groupKey) \
            .replace([np.inf, -np.inf], np.nan) \
            .dropna()

        df_numbins = df_org.groupby([groupKey, numBins]) \
            .size() \
            .reset_index()[[groupKey, numBins]]

        print("Adding", numBins, "column to the stat table ...")
        df_stat = df_stat.merge(df_numbins, on=groupKey).drop_duplicates()
        cols = df_stat.columns.tolist()
        cols.remove(numBins)
        df_stat = df_stat.groupby(cols)[numBins].sum().reset_index()

        print("Returning the stat table ...")
        print("------------------------------------------- ")
        return df_stat.sort_values("mean", ascending=False)



    def compute_schock_list(self, df_stat, medSeuil=20.):
        """ An outlier is defined as if the max count is larger than median count times medSeuil

        Parameters
        ----------
        df_stat : dataframe
            table of statistical indicators ( compute_stat )
        medSeuil : float
            A shock in bincount is detected if binCount > medSeuil * median

        Return
        ----------
         List of organName containing the outliers
        """

        print("---> Calling ExtractShockList : Extracts the organisations containing outliers.")
        print("    The output is sorted by mean count descending.")
        df_stat = df_stat[["organName", "median", "max", "mean"]].copy()
        df_stat["r"] = (df_stat["max"] / df_stat["median"]).replace(np.inf, np.nan)

        print("    Returning the outlier list ...")
        print("------------------------------------------- ")
        return df_stat.loc[df_stat["r"] > medSeuil] \
            .sort_values("mean", ascending=False)["organName"] \
            .tolist()


    def get_most_var_list(self, df_stat, minNumBins=6, meanCount=20):
        """ Gives the first 120 organisation containing the most variabilities.
            The larger the sigmasRatio = sugma/intenalSigma is the more variable a time serie will be.

        Parameters
        ----------
        df_stat : dataframe
            table of statistical indicators ( compute_stat )
        minNumBins : int
            Minimum number of bins needed for statistical computations
        meanCount : int
            Minimum average accepted number of citations (binCount).

        Return
        ----------
         List of the first 120 organName containing the most variabilities
        """

        print("---> Calling get_most_var_list to get list of most variable curves")
        print("    Minimum num of bins set to ", minNumBins)
        print("    Average count set to larger than ", meanCount)

        mostVars = df_stat.sort_values("sigmasRatio", ascending=False)
        cond = ((mostVars["numBins"] > minNumBins) &
                (mostVars["mean"] > meanCount)
                )
        organVarList = mostVars \
                           .loc[cond, "organName"] \
                           .unique() \
                           .tolist()[:120]
        print("    Returning the most variable list ...")
        print("------------------------------------------- ")
        return organVarList


    def get_leader_list(self, df_stat, nLeader=120):
        """ Computes the list of first nLeader organisations.

        Parameters
        ----------
        df_stat : dataframe
            table of statistical indicators ( compute_stat )
        nLeaders : int
            Size of the list containing the first most cited organisations.

        Return
        ----------
         List of organName sorted by mean number of citations
        """
        return df_stat["organName"].tolist()[:nLeader]

    ##############################################################################

