import sys
import argparse

sys.path.insert(0, './app/analysis')

from Model import Model

if __name__ == '__main__':

    # Set parser
    parser = argparse.ArgumentParser()

    #parser.add_argument('--gather', type=str, help='Gather Data From Symbol Pair (Default = )')
    parser.add_argument('--pair_id', type=str, help='Indicate pair to gather data from (Default = ETHBTC)', default='ETHBTC')
    parser.add_argument('--test', type=bool)

    ## PATH TO FILE FROM BASE ##
    parser.add_argument('--path', type=str, help='Path From Base Directory To File To Be Formatted', required=True)


    # DEBUG OPTION
    parser.add_argument('--debug', help='Debug True/False if set --debug flag, will output all messages every "--wait_time" ',
                        action="store_true", default=False) # 0=True, 1=False

    option = parser.parse_args()

    m = Model(option)