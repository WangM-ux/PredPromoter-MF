import argparse
import os
from codes.getModel import getResults

datapath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
dataPath = os.path.abspath(os.path.dirname(os.getcwd()))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="Accept fasta file in response to user")
    parser.add_argument("--fasta", required=True, help="input fasta file")
    args = parser.parse_args()
    input = args.fasta

    id, result = getResults(input)
    for i in range(len(id)):
        print(id[i].replace('\n', '') + " : " + result[i])






