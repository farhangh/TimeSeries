import numpy as np
from scipy.special import ndtr
from bb8TSA.TSA.dataBinModules import BinnedOrganisations, stack_c_DFs
from bb8TSA.TSA.statModules import StatIndic
from bb8TSA.TSA.tendanceModules import Tendance, constrained_tendance

class TimeSeriesAnalysis( BinnedOrganisations, StatIndic, Tendance ):
    """Time Series Analysis Class.

     Parameters
     ----------
     BinnedOrganisations : class
        Time series construction class.
        Time series are kept in a dataframe with 3 columns : organName, date and binCount
        Provides methods to bin the data according to given time-bin (freq)
        Each bin contains the citation count for a given organisation.

     StatIndic : class
         Provides methods to extract statistical indicators for time series extracted by
         BinnedOrganisations class.

    Tendance : class
        Provides methods to determine increase/decrease tendency in citations for the time series


    nameCol : str
        Column containing the name of the organisations

    dateCol : str
       Column containing the measurement date

    countCol : str, optional
        Column containing the wait parameter if exist.

    countFlag : Boolean
        True if countCol is included. False by default


    freq : str
        The bin size in weeks or a month {"W", "2W", "3W", "M"}
        default is month ("M")

    groupKey : str
        For grouping by organisation name. Keep the default value.

    targetCol : str
        Target column for aggregations. Keep the default value.

    intp : Boolean
        True by default to include interpolation to compute the variability of a time serie.

    numBins : str
        Name for the the column to count the total number of bins for each time serie

    medSeuil : float
        A factor to define the threshold to detect an outlier count (shock).
        outlier > medSeuil * median

    minNumBins : int
        Minimum number of bins needed for statistical computations

    meanCount : int
        Minimum average accepted number of citations.

    nLeader : int
        Size of the list containing the first most cited organisations.

    simpFit : boolean
        True to fit a line to time series excluding the shot noises, False otherwise

    noiseRatio : float
        This parameter is used to constrain the evident tendencies.
        If the ratio between the slope error and the slope of the fitted line are more
        than the noisRatio the time serie is excluded.

    memSeuil : float
        Maximum memory to be used to load all dataframes in DFs

    Attributes
    ----------
    tendance_info : dataframe
        Contains the name of the organisation, the slope and slope error of the fitted line to the
        time serie and the p_value with horizontal line as the null-hypothesis.

     """

    def __init__(self, nameCol="organName", dateCol="date", countCol=None, countFlag=False,
                 freq="M", groupKey="organName", targetCol="binCount", intp=True,
                 numBins="numBins", medSeuil=20., minNumBins=6, meanCount=20, nLeader=120, simpFit=False,
                 noiseRatio=.5, memSeuil=5.):
        print("TimeSeriesAnalysis class initialised.")
        self.groupKey = groupKey
        self.nameCol= nameCol
        self.dateCol = dateCol
        self.countCol = countCol
        self.targetCol = targetCol
        self.noiseRatio = noiseRatio
        self.simpFit = simpFit
        self.intp = intp
        self.numBins = numBins
        self.freq = freq
        self.minNumBins = minNumBins
        self.meanCount = meanCount
        self.countFlag = countFlag
        self.medSeuil = medSeuil
        self.nLeader = nLeader
        self.memSeuil = memSeuil

        print("TimeSeriesAnalysis class initialised.")
        BinnedOrganisations.__init__(self, nameCol, dateCol, countCol, freq, minNumBins, countFlag)
        StatIndic.__init__(self)
        Tendance.__init__(self)


    def fit(self, DFs, fileList=None):
        """ Extracts statistical information from the input dataframes
        Parameters
        ----------
        DFs : list of dataframes
            Dataframes read from different files sources.
            Each dataframes should have at least 2 columns : name and date

        fileList : list of str, optional
            List containing the name of the source files of the data frames with the same
            order as DFs.

        """

        self.load_DFs(DFs, fileList)
        # (BinnedOrganisations metheod)

        if self.countFlag :
            for i, df in enumerate(self._organ_DFs) :
                if self.countCol == None :
                    raise KeyError( "countCol name not specified." )
                elif self.countCol not in self._organ_DFs[i].columns :
                    raise KeyError("{} not found in schema.".format(self.countCol))

                self._organ_DFs[i] = self._organ_DFs[i][[self.nameCol, self.dateCol, self.countCol]].copy()
                self._organ_DFs[i].columns = ["organName", "date", "count"]
        else :
            for i, df in enumerate(self._organ_DFs):
                self._organ_DFs[i] = self._organ_DFs[i][[self.nameCol, self.dateCol]].copy()
                self._organ_DFs[i].columns = ["organName", "date"]

        fit_info = self.compute_tendance_DFs()[ ["organName", "mean", "median", "sigmasRatio", "intercept",
                                                "slope", "slopeErr", "max", "linePValue", "numBins"] ]\
            if self.simpFit \
            else self.compute_tendance_DFs()[ ["organName", "mean", "median", "sigmasRatio",
                                                "slope", "slopeErr", "max", "numBins", "intercept"] ]


        fit_info["citation_rank"] = range(1, len(fit_info)+1)

        shock_list = self.compute_schock_list(fit_info, medSeuil=self.medSeuil)
        fit_info["count_shoot"] = "No"
        fit_info.loc[ fit_info["organName"].isin(shock_list), "count_shoot" ] = "Yes"

        fit_info["nbinR"] = (fit_info["numBins"]-1.)/fit_info["numBins"]
        fit_info["variability"] = ""
        fit_info.loc[fit_info["sigmasRatio"]<=1, "variability"]="Low"
        fit_info.loc[fit_info["sigmasRatio"]<=fit_info["nbinR"], "variability"]="None"
        fit_info.loc[fit_info["sigmasRatio"]>1, "variability"]="Moderate"
        fit_info.loc[fit_info["sigmasRatio"]>=2, "variability"]="high"

        fit_info["tendency"] = "Const"
        fit_info.loc[fit_info["slope"]<-1.e-6, "tendency"] = "Decrease"
        fit_info.loc[fit_info["slope"]>1.e-6, "tendency"] = "Increase"

        #fit_info["p_0"] = round(fit_info["linePValue"], 2)

        def compute_p_0(row) :
            p_0 = ndtr(-row[0]/row[1]) if row[0]>0 else 1.- ndtr(-row[0]/row[1])
            return p_0
        fit_info["p_0"] = fit_info[["slope", "slopeErr"]].apply(compute_p_0, axis=1)
        fit_info["p_0"] = round(fit_info["p_0"], 2)
        fit_info.loc[ (abs(fit_info["slope"])<1.e-6) & (fit_info["slopeErr"].isnull().values.any()), "p_0"] = 1.

        self.tendance_info = fit_info[["organName", "slope", "slopeErr", "intercept", "p_0"]]

        fit_info.rename(columns={"mean": "mean_citation"}, inplace=True)

        self._fit_info = fit_info[["organName", "mean_citation", "citation_rank", "count_shoot", "variability",
                        "tendency", "p_0", "numBins"]]


    def transform(self) :
        """
        :return: dataframe
            A dataframe showing the citation rank and tendency of an organisation and whether
            the corresponding time serie shows variability and/or shock count.

        """
        return self._fit_info

    def fit_transform(self, DFs, fileList=None):
        """ Performes fit and transform provided by the TimeSeriesAnalysis class.
        :param DFs: list of dataframes
            combined time series.
        :fileList: list of str, optional
        :return: dataframe
        """
        self.fit(DFs, fileList)
        return self.transform()


    def compute_tendance_DFs(self):
        """ Computes the tendency of the time series extracted from input dataframes.
        (tendanceModules.py)

        return
        ------
        The tendency dataframe
        """

        df_c = self.stackDFs()
        statTable = self.compute_stat(df_c, groupKey=self.groupKey, targetCol=self.targetCol,
                                      intp=self.intp, numBins=self.numBins)
        df_tendance = self.compute_tendance(df_c, statTable, freq=self.freq, simpFit=self.simpFit)
        return df_tendance

    def compute_stat_DFs(self) :
        """ Computes the statistical indicators of the time series extracted from input dataframes.
        (statModules.py)

        return
        ------
        Table of statistics including the variability parameter sigmasRatio.
        """
        df_c = self.stackDFs()
        statTable = self.compute_stat(df_c, groupKey=self.groupKey, targetCol=self.targetCol,
                                      intp=self.intp, numBins=self.numBins)
        return statTable


    def get_df(self, DFs):
        """
        :param DFs: list of dataframes
        :return: dataframe
            cleaned binned dataframe
        """
        self.load_DFs(DFs)
        return self.stackDFs()


    def stackDFs(self):
        """ Stack the binned input dataframes. (dataBinModules.py)
        return
        ------
        The stacked dataframe
        """
        # Filtering the binned dataframes with number of bins > minNumBins
        c_dfs = self.extract_cleand_binned_DFs
        # Concating all binned dataframes
        df_c = stack_c_DFs(c_dfs, memSeuil=self.memSeuil)
        return df_c


    def constrained_tendance_DFs(self):
        """ Computes the constrained tendency with respect to the noiseRatio.
        (tendanceModules.py)

        return
        ------
        The constrained tendency dataframe
        """
        df_tendance = self.compute_tendance_DFs()
        return constrained_tendance(df_tendance, noiseRatio=self.noiseRatio, minNumBins=self.minNumBins)

    def compute_schock_list_DFs(self):
        """ Computes the list of organisation containing outliers (shocks) with respect to medSeuil
            (statModules.py)

        return
        ------
        List of organisation names
        """
        statTable = self.compute_stat_DFs()
        return self.compute_schock_list(statTable, self.medSeuil)

    def get_most_var_list_DFs(self):
        """ Computes the list of organisation containing most variations in time
            (statModules.py)

        return
        ------
        List of organisation names
        """
        statTable = self.compute_stat_DFs()
        return self.get_most_var_list(statTable, minNumBins=self.minNumBins, meanCount=self.meanCount)

    def get_leader_list_DFs(self):
        """ Computes the list of first nLeader organisations.
            (statModules.py)

        return
        ------
        List of organisation names
        """
        statTable = self.compute_stat_DFs()
        return self.get_leader_list(statTable, nLeader=self.nLeader)

