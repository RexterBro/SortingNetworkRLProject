from main import sort_with_network
import numpy as np
import sys

try:
    test_array = np.array(sys.argv[1].split(','), dtype=np.int8)
    verbose = bool(sys.argv[2].split(','))
    network = np.loadtxt('tester_network')
    print('loaded network')
    print('sorting array [' + str(test_array) + ']')
    sorted_array = sort_with_network(network, test_array, verbose)
    print('after sort: ' + str(sorted_array))

except:
    print('usage: python network_tester.py 1,2,3,4,5,6,7,8 false')
