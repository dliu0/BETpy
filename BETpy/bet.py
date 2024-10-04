import numpy as np
import math
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
from bitarray import bitarray
from joblib import Parallel, delayed

#get all functions into one package
class DDMbet:
    def __init__(self) -> None:
        pass
        
    def binary_expansion(self, number, precision=2):
        """
        Find binary representation of a number between 0 and 1 upto precision
        Revert 1 to 0 and 0 to 1 in the return binary array for future use
        Returns:
        Binary array
        """
        binary_representation = np.full(precision, 1, dtype=bool)
        if number < 0 or number > 1:
            raise ValueError("The number must be between 0 and 1.")
        else:
            index = 0
            while number > 0 and index < precision:
                number *= 2.0
                if number >= 1:
                    binary_representation[index]= 0
                    number -= 1
                index+=1
        #returns a bit array of size,precision,boolean type
        bin_arr =  bitarray()
        bin_arr.pack(binary_representation.tobytes())
        return bin_arr

    def binary_expansion_int(self, number, bits):
        """
        Find binary representation of an integer >=1  upto bits
        Reverse the order of bits, least significant bit first
        Returns:
        Int array of 0 and 1
        """
        if number >= 0:
            exp = [int(digit) for digit in bin(number)[2:].zfill(bits)]
            exp = exp[:bits]
            #reverse the digits order
            return exp[::-1]
        else:
            raise ValueError("The number must be >= 0.")
            
    def normalize_by_sorted_position(self, data):
        """
        Normalizes the input data by dividing the position of each value in the sorted list
        by the length of the list and then scaling to the range [0, 1].

        Parameters:
        data (list of float): The data to be normalized.

        Returns:
        list of float: The normalized data.
        """
        # Step 1: Sort the data and create a mapping from value to its sorted positions
        sorted_data = sorted(data)
        position_dict = {}
    
        # Assign unique positions to each value in the sorted list
        for index, value in enumerate(sorted_data):
            if value not in position_dict:
                position_dict[value] = []
            position_dict[value].append(index + 1)
    
        # Step 2: Assign positions to the original data
        positions = []
        used_positions = {key: 0 for key in position_dict}
        for value in data:
            position = position_dict[value][used_positions[value]]
            positions.append(position)
            used_positions[value] += 1
    
        # Step 3: Normalize each position by the length of the list
        list_length = len(data)
        normalized_positions = [(position / list_length) for position in positions]
    
        return normalized_positions

    #calculate block signs of one coordinate
    def calc_block_signs(self, cx):
        #binary interaction of one axis
        c_len = len(cx)
        max_index=-1
        #find smallest block size
        for i in range(c_len):
            if cx[i]:
                max_index = i

        if max_index >=0:
            #calculate block size and number of blocks
            blocksize = 2**(-1*(max_index+1))
            blocks = 2**(max_index+1)
            #calculate sign of each block
            block_signs=[]
            for i in range(blocks):
                #assign each block a initial sign based on the minimum block
                if i%2==1:
                    this_block_sign = -1
                else:
                    this_block_sign = 1
                #loop through rest of values before minimum block
                for k in range(0,max_index):
                    if cx[k]:
                        #number of minimum blocks that make up current block
                        num_b = 2**(max_index- k)
                        #current block index
                        new_b = i//num_b + 1
                        #find the sign of the current block
                        if new_b % 2 == 1:
                            #update current blocks sign
                            this_block_sign = this_block_sign * (-1)
                #update block sign list for graphing
                block_signs.append(this_block_sign)
            
            return blocksize, block_signs
        else:
            return 1, [1]
    #plot function for the data    
    def plot_cross(self, dataX, dataY, cross):
        """
        Plot the inputs, binary interaction design and results.

        Parameters:
        dataX: input array 1
        dataY: input array 2
        cross: binary interaction design
        """
        #defines plot that will be used
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #splits cross/binary interaction into x and y
        c_len = len(cross)//2
        #x is first half, y is second half
        cx = cross[0:c_len]
        cy = cross[c_len:]
        #calculate the signs of each block given interaction
        #used for coloring of the plot's sections
        #bzx/bxy = block size, sx/sy = least of block signs
        bzx, sx = self.calc_block_signs(cx)
        bzy, sy = self.calc_block_signs(cy)
        #normalize data x and data y
        normalized_data_x = self.normalize_by_sorted_position(dataX)
        normalized_data_y = self.normalize_by_sorted_position(dataY)
        #plot the normalized data into 0 to 1 plot
        ax1.scatter(normalized_data_x, normalized_data_y)
        
        #plotting colors for interaction on the normalized plot
        for i in range(len(sx)):
            #start x coordinate of block
            block_x_start = i*bzx
            #end x coordinate of block
            block_x_end = (i+1)*bzx
            
            #check each y block
            for j in range(len(sy)):
                #start y coordinate of block
                block_y_start = j*bzy
                #end y coordinate of block
                block_y_end = (j+1)*bzy
                #block sign based on x and y block signs
                sign = sx[i] * sy[j]
                #sign determines color of block, if sign 1, then color the block blue
                if sign == 1:
                    ax1.axvspan(block_x_start, block_x_end, ymin=block_y_start, ymax=block_y_end, facecolor='blue', alpha=0.3)
        #label this graph as normalized plot                
        ax1.set_title('Transformed')
        #plot the x and y values from orignal data, label this graph as original plot
        ax2.scatter(dataX, dataY)
        ax2.set_title('Original')

        plt.tight_layout()
        plt.show()
        
    def selective_product_binary(self, exp, binary_xy):
        #count the number of nonzero values in bit array, binary_xy & exp
        #represents number of -1s in the product of the interaction and binary expansion arrays
        count = (binary_xy & exp).count(1)
        #odd # of 1's indicate group 1, even # of 1's indicate group 0
        return count%2
    #calculate p-value of two tailed binomial distribution
    def binomial_p_value_two_tailed(self, n, k, p=0.5):
        # Two-tailed p-value
        p_value = binom.cdf(k, n, p)
        if p_value > 0.5:
            p_value = 1.0 - p_value
        return 2 * p_value

    def calcBET(self, bin_xy, find_min = False, max_depth=8, p_value_threshold=0.05):
        #find depth of interaction
        num_points= len(bin_xy)
        N = min(int(math.log2(num_points)//1 + 1), max_depth+1)
        depth = min(N, max_depth)
        #assign value for binary interaction depth, how many interactions will be checked
        num_search = 2**(depth+depth)
        #initial values
        max_diff=0
        mincount=0
        min_cross=[]
        pval_factor = (2**N-1)*(2**N-1)
        #check interactions from 1 to assigned depth
        #searches/loops through all interactions between group 0 and group 1
        for com in range(1, num_search+1):
            #track points in group 0
            g_count=0
            #generate interaction vector
            exp = self.binary_expansion_int(com, 2*N)
            exp_x = exp[::2]
            exp_y = exp[1::2]
            #combine x's and y's into 1 interaction array
            exp = np.concatenate([exp_x,exp_y])
            #convert array into bit array
            exp_bool = np.array(exp, dtype=bool)
            cb_exp =  bitarray()
            cb_exp.pack(exp_bool.tobytes())
            #counting the number of points in group 0
            g_count = sum([self.selective_product_binary(cb_exp, num) for num in bin_xy])
            #find which group has lower count of numbers
            lower_count = min(g_count, num_points-g_count)
            #track minimum p-value by tracking the maximum difference between groups
            if abs(num_points-g_count-g_count) > max_diff:
                #find p-value of the specific interaction
                pv = self.binomial_p_value_two_tailed(num_points, lower_count, 0.5) * pval_factor
                min_p =pv
                mincount = lower_count
                min_cross = exp
                max_diff = abs(num_points-g_count-g_count)
                 #will stop looking once minimum p-value is less than p-value threshold (0.05 in this case), and if the find_min is false
                if min_p < p_value_threshold and find_min == False:
                    break
        return min_p, mincount, min_cross

    def runBET(self, dataX, dataY, plot=False, find_min = False, max_depth=8, p_value_threshold=0.05):
        """
        Compute BET statistics, p-value and Binary interaction design for two input variables

        Parameters:
        dataX: input array 1
        dataY: input array 2
        plot: Boolean. whether to plot the input and results. Default=False
        find_min: Boolean. whether to find the binary interaction that produces the lowest p-value, Default=False, 
                  return the first binary interaction that produces the p-value less than p_value_threshold
        p_value_threshold: dependency test p-value threshold. default = 0.05.

        Returns:
        p-value, count of one group, binary interaction design
        """
        #normalize x and y data inputted
        normalized_data_x = self.normalize_by_sorted_position(dataX)
        normalized_data_y = self.normalize_by_sorted_position(dataY)
        #determine the depth for binary expansion, Dx = Dy and Dx = logbase2(len(x))
        num_points= len(normalized_data_x)
        N = min(int(math.log2(num_points)//1 + 1), max_depth+1)
        #combine binary expansions of x and y into expansion list (exp_xy), used to find interactions
        exp_xy=[]
        for x,y in zip(normalized_data_x, normalized_data_y):
            expval = self.binary_expansion(x, N) + self.binary_expansion(y, N)
            exp_xy.append(expval)
        #run calculation function to iterate through all binary interactions to depth
        min_p, mincount, min_cross = self.calcBET(exp_xy, find_min=find_min, max_depth=max_depth, p_value_threshold=p_value_threshold)
        if min_p > p_value_threshold:
            min_cross = np.zeros(len(min_cross))
        if plot:
            self.plot_cross(dataX, dataY, min_cross)
        #returns minimum p-value, minimum group count, interaction of minimum p-value    
        return  min_p, mincount, min_cross
    
    def test_variable_pairs(self,indf, max_depth=2,nprocess=1):
        """
        Compute BET statistics, p-value and Binary interaction design for all pairs of columns in the input dataframe

        Parameters:
        indf: input dataframe
        max_depth: Maximium depth of the binary expansion. Default=2
        nprocess: Number of processes to use. Default=1. If is -1, use the number of cpus

        Returns:
        dataframe: results for each pair of columns
        """
        columns = indf.columns
        results = pd.DataFrame(columns=['VarX', 'VarY', 'P-val', 'Min Count', 'Min Cross'])
        n = len(columns)

        def compute_pairwise_bet(i,j):
            colx = columns[i]
            coly = columns[j]
            min_p, mincount, min_cross = self.runBET(indf[f'{colx}'],indf[f'{coly}'], find_min=False, plot=False, max_depth=max_depth, p_value_threshold=0.05)
            new_row = {'VarX':colx, 'VarY':coly, 'P-val':min_p, 'Min Count':mincount, 'Min Cross':min_cross}
            return new_row

        results = Parallel(n_jobs=nprocess)(
            delayed(compute_pairwise_bet)(i, j)
            for i in range(n) for j in range(i+1, n))
        
        results_df = pd.DataFrame(results)
        return results_df


#test
if __name__ == "__main__":
    angle = np.random.uniform(0, 2*math.pi, size=1000)
    dataX = [math.sin(x) for x in angle]
    dataY = [math.cos(x) for x in angle]
    ddmbet = DDMbet()
    min_p, mincount, min_cross= ddmbet.runBET(dataX, dataY)

    print(f"Minimum P-value: {min_p}")
    print(f"Group 0 count: {mincount}")
    print(f"Cross action: {min_cross}")