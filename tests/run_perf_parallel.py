import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from BETpy.bet import DDMbet
import time
import math
from joblib import Parallel, delayed

def runtest(indf, run_i):
    columns = indf.columns
    results = pd.DataFrame(columns=['VarX', 'VarY', 'P-val', 'Min Count', 'Min Cross'])
    start_time = time.time()
    count=0
    ddmbet = DDMbet()
    n = len(columns)

    def compute_pairwise_bet(i,j):
        colx = columns[i]
        coly = columns[j]
        min_p, mincount, min_cross = ddmbet.runBET(indf[f'{colx}'],indf[f'{coly}'], find_min=False, plot=False, max_depth=2, p_value_threshold=0.05)
        new_row = {'VarX':colx, 'VarY':coly, 'P-val':min_p, 'Min Count':mincount, 'Min Cross':min_cross}
        return new_row

    results = Parallel(n_jobs=-1)(
        delayed(compute_pairwise_bet)(i, j)
        for i in range(n) for j in range(i+1, n))

    end_time = time.time()
    dur = end_time - start_time
    print(f"Execution time {count}: {dur:.5f} seconds")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'c:\\temp\\results_out_{run_i}.csv', index=False)
    indf.to_csv(f'c:\\temp\\inputdata_{run_i}.csv', index=False)


mydf=pd.DataFrame()
dataX = np.random.uniform(0, 20*math.pi, size=140)
mydf[f"col_0"] =dataX
for i in range(1,100):
   
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY
runtest(mydf,1)

mydf=pd.DataFrame()
dataX = np.random.uniform(0, 20*math.pi, size=140)
mydf[f"col_0"] =dataX
for i in range(1,100):
   
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY
runtest(mydf,2)

mydf=pd.DataFrame()
dataX = np.random.uniform(0, 20*math.pi, size=140)
mydf[f"col_0"] =dataX
for i in range(1,100):
   
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY
runtest(mydf,3)

mydf=pd.DataFrame()
dataX = np.random.uniform(0, 20*math.pi, size=140)
mydf[f"col_0"] =dataX
for i in range(1,100):
   
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY
runtest(mydf,4)

mydf=pd.DataFrame()
dataX = np.random.uniform(0, 20*math.pi, size=140)
mydf[f"col_0"] =dataX
for i in range(1,100):
   
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY
runtest(mydf,5)
