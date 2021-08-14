from bb8TSA.TSA.tsaModules import TimeSeriesAnalysis

def some_tests( DFs, fileList ) :
    tsa =TimeSeriesAnalysis( countFlag=True, countCol="count" )
    tsa.fit(DFs, fileList)
    assert(tsa.get_leader_list_DFs()[:3] == ['ue', 'covid19', 'oms'])
    assert(tsa.get_most_var_list_DFs()[:4] == ['granddébatnational', 'granddébat', 'oms', 'pge'])
    assert(tsa.compute_schock_list_DFs()[:4] == ['ab', 'charliehebdo', 'pge', 'cheminots'])
