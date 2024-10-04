import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from BETpy.bet import DDMbet
import time
import math

def runtest(indf):
    columns = indf.columns
    results = pd.DataFrame(columns=['VarX', 'VarY', 'P-val', 'Min Count', 'Min Cross'])
    start_time = time.time()
    count=0
    ddmbet = DDMbet()
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            count+=1
            colx = columns[i]
            coly = columns[j]
            min_p, mincount, min_cross = ddmbet.runBET(indf[f'{colx}'],indf[f'{coly}'], find_min=False, plot=False, max_depth=2, p_value_threshold=0.05)
            new_row = {'VarX':colx, 'VarY':coly, 'P-val':min_p, 'Min Count':mincount, 'Min Cross':min_cross}
            results.loc[len(results)] = new_row

    end_time = time.time()
    dur = end_time - start_time
    print(f"Execution time {count}: {dur:.5f} seconds")
    results.to_csv('c:\\temp\\results_out.csv', index=False)
    indf.to_csv('c:\\temp\\inputdata.csv', index=False)


mydf=pd.DataFrame()
dataX = np.random.uniform(0, 20*math.pi, size=100)
mydf[f"col_0"] =dataX
for i in range(1,100):
   
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY

runtest(mydf)