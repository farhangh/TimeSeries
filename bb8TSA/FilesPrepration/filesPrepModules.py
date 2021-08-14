from os import path
import sys

import pandas as pd


def extract_file_list(fileList, minDays=89, memSeuil=5.):
    """ Parses the input data files from the console and verify if they are the right files
    :param fileList: list of str
        liste of files read from the console.
    :param minDays: int
        minimum accepted number of days in each data file.
    :param memSeuil: float
        maximum dedicated memory in GB.
    :return: list of accepted data file names
    """


    # Verifying the availability of the input files and the data quantity
    # minDays : minimum accepted number of days in the dataset
    # memSeuil : maximum memory in GB available to keep all dataframes

    print("extract_file_list : initial file list :", fileList)
    print("    Minimum accepted number of days: ", minDays)
    print("    Maximum dedicated memory: ", memSeuil, "GB")

    try:
        rem = []
        for file in fileList:
            if not path.isfile(file):
                print("extract_file_list : Warning: " + file +\
                      "is not available and will be dropped from the file list.")
                rem.append(file)

        fileList2 = list(set(fileList) - set(rem))

        if len(fileList2) == 0:
            print("extract_file_list : Error : No valid file(s). Exiting ...")
            sys.exit()

        # Computing the memory used by all dataframes in GB
        memUseList = list(map(lambda x: pd.read_parquet(x)\
                              .memory_usage(index=True).sum() * 1.e-9, fileList2))
        memUse = sum(memUseList)
        if memUse > memSeuil:
            raise MemoryError("extract_file_list : Files' volume ({} GB) exceeds the dedicated memory." \
                              .format(round(memUse, 1)))

        # compute the time length for each file
        date_DFs = list(map(lambda x: pd.read_parquet(x, columns=["date"]), fileList2))
        delDate = list(map(lambda dfDate, file:
                           (file, (pd.to_datetime(dfDate["date"]).max() -
                                   pd.to_datetime(dfDate["date"]).min()).days),
                           date_DFs, fileList2
                           )
                       )

        fileList3 = list(map(lambda tup: tup[0] if tup[1] > minDays else None, delDate))
        fileList3 = list(filter(None.__ne__, fileList3))

        if len(fileList3) < len(fileList2):
            print("extract_file_list : Files with data less than ", minDays, " days are discarded:",
                  set(fileList2) - set(fileList3))

        if len(fileList3) == 0:
            print("extract_file_list : Dataset(s) are not long enough. Exiting ...")
            sys.exit()

        print("extract_file_list : List of available files : ", fileList3)

        print("Returning the final file list ...")
        print("------------------------------------------- ")
        return fileList3

    except KeyError:
        print("extract_file_list : Error : No 'date' column found. Exiting ...")
        sys.exit()

##################################################################################

class FileNormalisation:
    """ Renrmalising the columns of the input data files

    Parameters
    ----------
    fileList : list of str
        list of data file names
    """

    def __init__(self, fileList):
        print("FileNormalisation class initialised.")
        self._fileList = fileList
        self._DFs = list(map(lambda x: pd.read_parquet(x), fileList))

    def get_initial_DFs(self):
        """
        :return: list of dataframes read from the data files
        """
        return self._DFs

    def make_normal(self, df, fname):
        """ Some renormalisation steps :
        # --Converting date type from string to datetime.
        # --Putting the 'name' column in lower case.
        # --Discarding non-alphanumerics from the 'name' column.
        # --Combining same organisation appeared with different names.

        :param df: dataframe
            dataframe read from a data file
        :param fname: str
            data file name
        :return: dataframe
        """
        print("file_normalisation : make_normal : Normalising", fname, "...")

        df = df[df["variable"] == "organisation"].copy()

        df["name"] = df["name"].apply(lambda x: x.lower()) \
            .apply(lambda x: "".join(filter(str.isalnum, x)))

        df.loc[df.name == "bpi", "name"] = "bpifrance"
        df.loc[df.name == "uniondesmétiersetdesindustriesdelhôtellerie", "name"] = "umih"
        df.loc[df.name.isin(["covid", "corona", "coronavirus"]), "name"] = "covid19"
        df.loc[df.name == "giletsjaune", "name"] = "giletsjaunes"
        df.loc[df.name == "organisationmondialedelasanté", "name"] = "oms"
        df.loc[df.name == "unioneuropéenne", "name"] = "ue"

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date"])

        return df

    def get_Normal_DFs(self):
        """
        :return: list of normalised dataframes
        """
        return list(map(self.make_normal, self._DFs, self._fileList))


    def get_NormalReduced_DFs(self):
        """
        :return: list of normalised dataframes each of which contains only 3 columns
        """
        normDfs = list(map(self.make_normal, self._DFs, self._fileList))
        for df in normDfs :
            df.rename(columns={"name": "organName"}, inplace=True)
            df = df[["organName", "date", "count"]].copy()
        return normDfs

