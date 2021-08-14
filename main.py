# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:20:39 2020

@author: M77100
F HABIBI Aban-Azar 99
"""

import sys
import pandas as pd
from time import time

from bb8TSA.FilesPrepration.filesPrepModules import extract_file_list, FileNormalisation
from bb8TSA.TSA.tsaModules import TimeSeriesAnalysis
#from test.test import *

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 20)

#'../Data/df_ner_long_2019.parquet ../Data/df_ner_long_2020.parquet ../Data/test.parquet nofile.parquet'

t0 = time()
if __name__=="__main__" :

    # Verifying the availability of the input files 
    file_list = extract_file_list( sys.argv[1:] )
    #file_list = ["C:/Users/M77100/Work/bpi-fr-data-sc-bb8-ts-analysis/test2.parquet"]
    # Normalising the input datasets
    # Making an instance of file_normalisation
    fn = FileNormalisation( file_list )
    dfs = fn.get_initial_DFs()
    normal_dfs = fn.get_Normal_DFs()
    normalR_dfs = fn.get_NormalReduced_DFs()

    tsa =TimeSeriesAnalysis( noiseRatio=.5, countFlag=True, countCol="count" )
#    tsa.fit(normalR_dfs, fileList=file_list)
#    tsa.fit(normalR_dfs)
#    df_info = tsa.transform()
    df_info = tsa.fit_transform(normalR_dfs)
#    print( "main : tendency : \n", df_info)
    print( "main : tendency : \n",
           df_info[:30].drop(columns=["mean_citation", "citation_rank"]) )


#    print(tsa.get_leader_list_DFs())
#    print(tsa.get_most_var_list_DFs())
#    print(tsa.compute_schock_list_DFs())
#    print(tsa.constrained_tendance_DFs())
#    print(tsa.stackDFs())


#    some_tests(normal_dfs, fileList=file_list)



print( "Time elapsed : ", round( time()-t0, 1 ), "s" )