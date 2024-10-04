import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from BETpy.bet import DDMbet
import time
import math

ddmbet = DDMbet()
mydf=pd.DataFrame()
np.random.seed(0)
dataX = np.random.uniform(0, 20*math.pi, size=200)
mydf[f"col_0"] =dataX
for i in range(1,4):
    dataY =[math.sin(0.3*i*x) for x in dataX]
    mydf[f"col_{i}"] =dataY
start_time = time.time()
results_df = ddmbet.test_variable_pairs(mydf, max_depth=8,nprocess=-1)

print(results_df)
end_time = time.time()
total_time = end_time - start_time  # Calculate the total time
print(f"\033[1mTotal execution time for all pairs: {total_time:.4f} seconds\033[0m")
