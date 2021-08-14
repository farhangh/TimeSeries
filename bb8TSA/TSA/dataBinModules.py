# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:06:33 2020

@author: M77100
"""
import datetime
import pandas as pd


class BinnedOrganisations:
    """Time series construction class.
        Time series are kept in a dataframe with 3 columns : organName, date and binCount
        Provides methods to bin the data according to given time-bin (freq)
        Each bin contains the citation count for a given organisation.

    Parameters
    ----------
    nameCol : str
        Column containing the name of the organisations

    dateCol : str
       Column containing the measurement date

    countCol : str, optional
        Column containing the wait parameter if exist.
    freq : str
        The bin size in weeks or a month {"W", "2W", "3W", "M"}. default is month ("M")

    minNumBins : int
        Minimum number of bins needed for statistical computations

    Attributes
    ----------
    extract_binned_DFs : list of dataframes
        for each input dataframe computes the corresponding time series and keeps it in a dataframe.

    extract_cleand_binned_DFs : list of dataframes
        counts the number of measurement (bins) of all times series and drop those with low number
        of bins according to minNumBins value.

    timeBinWidth : str
        adjusting the time width bins either a month(M) or 1,2,3 weeks (W, 2W, 3W).

    minNB : int
        Minimum number of bins needed for statistical computations.

    countFlag : Boolean
        True if countCol is included. False by default

    """

#    def __init__(self, nameCol="organName", dateCol="date", countCol=None, freq="M",
#                 minNumBins=3, countFlag=False):
    def __init__(self, nameCol, dateCol, countCol, freq, minNumBins, countFlag):

        print("BinnedOrganisations class initialised.")
        offsets = {"W": 2, "2W": 0, "3W": 0, "M": 27}
        self.minNumBins = minNumBins
        self.nameCol= nameCol
        self.dateCol = dateCol
        self.countCol = countCol
        self.countFlag = countFlag
        self._offsets = offsets
        self.freq = freq if freq in offsets.keys() else "NV"
        if self.freq == "NV" :
            raise ValueError("Time frequency should be one of :", self._offsets.keys())

        print("BinnedOrganisations : Allowd time frequencies are :", offsets.keys())
        print("BinnedOrganisations : time frequency set to ", self.freq)


    def load_DFs(self, DFs, fileList=None):
        """
        :param DFs: list of initial raw dataframes
        :param fileList: name of corresponding data source files
        :return: keep these parameters as class private attributes
        """
        _fileList = fileList if fileList is not None \
            else [ "file"+str(i+1) for i in range( len(DFs) ) ]
        self._organ_DFs = DFs
        self._fileList = _fileList

    def make_binned_time_series(self, df_organs, fname):
        """ Computes the time series

        Parameters
        ----------
        df_organs : dataframe
            it should contain at least two columns of name and date (timestamp)
        fname : string
            original data file name

        Return
        ----------
        A time series dataframe according to a given time bin
        """

        print("BinnedOrganisations : MakeBinnedTimeSeris : Binning on time for ", fname, ".")
        freq = self.freq
        print("    Binning frequency : ", freq)
        print("    Grouping by orgnanisations' name and time  binned by {} then sum over".format(freq),
                    "the binn's count ...")

        df_organs.rename(columns={self.nameCol:"organName", self.dateCol:"date"}, inplace=True)
        if self.countFlag == False:
            df_organs["count"] = 1
        else :
            df_organs.rename(columns={self.countCol: "count"}, inplace=True)

        df_organ_binned = df_organs.groupby(["organName", pd.Grouper(key='date', freq=freq)]) \
            .agg({"count": "sum"}) \
            .reset_index()

        # Adjusting the date bin centers to the first day of the month
        dd = self._offsets[self.freq]
        df_organ_binned["date"] = df_organ_binned["date"] \
            .apply(lambda x: x - datetime.timedelta(dd))
        #            .apply( lambda x : x.replace(day=1) )

        print("    Renaming the columns of the binned datafarme ...")
        df_organ_binned.columns = ["organName", "date", "binCount"]
        # binCount : number of enteries per bin

        print("    Returning the binned datafarme ...")
        print("------------------------------------------- ")
        return df_organ_binned


    def make_clean_cuts(self, df_organ_binned, fname):
        """ Counts the number of measurement (bins) of all times series and drop those with low number
        of bins according to minNumBins value.

        Parameters
        ----------
        df_organ_binned : dataframe
            time seroes computed by make_binned_time_series
        fname : string
            original data file name

        Return
        ----------
        dataframe
        """

        print("BinnedOrganisations : make_clean_cuts : Filtering low count curves for ", fname, ".")
        print("    Minimum accepted number of bins : ", self.minNumBins)

        df_c = df_organ_binned.groupby("organName")["date"] \
            .count() \
            .reset_index()
        df_c.rename(columns={"date": "numBins"}, inplace=True)
        # numBins : number of bins for an organisation

        df_c = df_c[df_c["numBins"] > self.minNumBins - 1]
        df_c = df_organ_binned.merge(df_c, on="organName")

        print("    Returning the cleaned binned datafarme ...")
        print("------------------------------------------- ")

        return df_c

    def get_mcc(self):
        """
        Return
        ----------
        List of dataframes each of which contains cleaned time series
        """
        binned_DFs = self.get_mbts()
        return list(map(self.make_clean_cuts, binned_DFs, self._fileList))

    extract_cleand_binned_DFs = property(get_mcc)

    def get_mbts(self):
        """
        Return
        ----------
        List of dataframes each of which contains time series
        """

        organ_DFs = self._organ_DFs
        binned_DFs = list(map(self.make_binned_time_series, organ_DFs, self._fileList))
        self._binned_DFs = binned_DFs
        return binned_DFs

    extract_binned_DFs = property(get_mbts)

    def get_freq(self):
        return self.freq

    def set_freq(self, val):
        if val not in self._offsets.keys():
            raise ValueError("Time frequency should be one of :", self._offsets.keys())
        else:
            self.freq = val

    timeBinWidth = property(get_freq, set_freq)

    def get_minNumBins(self):
        return self.Bins

    def set_minNumBins(self, val):
        if val < 2:
            raise ValueError("Minimum number of bins should be larger than 1")
        else:
            self.minNumBins = val

    minNB = property(get_minNumBins, set_minNumBins)


##############################################################################


def stack_c_DFs(c_DFs, memSeuil=5):
    # Computing the memory used by all dataframes in GB
    # with memSeuil you can manage your memory resource usage
    memUseList = list(
        map(lambda df: df.memory_usage(index=True).sum() * 1.e-9, c_DFs)
    )
    memUse = sum(memUseList)
    print("stackDFs : Total memory used by dataframes :", memUse, "GB")

    df_c = pd.DataFrame()
    if memUse < memSeuil:
        for df in c_DFs:
            df_c = pd.concat([df_c, df], ignore_index=True)\
                     .sort_values(by=["organName", "date"])
        df = df_c[["organName", "numBins"]]\
                        .drop_duplicates()\
                        .groupby("organName")\
                        .agg( {"numBins":"sum"} )
        df_c.drop( columns=["numBins"], inplace=True )
        df_c = df_c.merge(df, on="organName")

    else:
        raise MemoryError("stackDFs : Too big dataframe to concat. Exiting ...")

    return df_c

##############################################################################