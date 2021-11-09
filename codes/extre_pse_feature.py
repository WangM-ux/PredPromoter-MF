import itertools
import os
import pickle
import re
import sys

import numpy as np


baseSymbol = 'ACGT'
myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
}

myDictDefault = {
    'DAC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
            'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'DCC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
            'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'DACC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
             'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'TAC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'TCC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'TACC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'PseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},

    '90PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist',
                       'Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content', 'A-philicity',
                       'Propeller twist', 'Duplex stability-free energy', 'Duplex stability-disrupt energy', 'DNA denaturation', 'Bending stiffness',
                       'Protein DNA twist',  'Stabilising energy of Z-DNA',  'Aida_BA_transition',  'Breslauer_dG',  'Breslauer_dH',
                       'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition', 'Ivanov_BA_transition',
                       'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG', 'SantaLucia_dH', 'SantaLucia_dS',
                       'Sarai_flexibility', 'Stability', 'Stacking_energy', 'Sugimoto_dG', 'Sugimoto_dH',
                       'Sugimoto_dS', 'Watson-Crick_interaction', 'Stacking energy', 'Bend', 'Tip',
                       'Inclination', 'Major Groove Width', 'Major Groove Depth', 'Major Groove Size', 'Major Groove Distance',
                       'Minor Groove Width', 'Minor Groove Depth', 'Minor Groove Size', 'Minor Groove Distance', 'Persistance Length',
                       'Melting Temperature', 'Mobility to bend towards major groove', 'Mobility to bend towards minor groove', 'Propeller Twist', 'Clash Strength',
                       'Enthalpy', 'Free energy', 'Twist_twist', 'Tilt_tilt', 'Roll_roll',
                       'Twist_tilt', 'Twist_roll', 'Tilt_roll', 'Shift_shift', 'Slide_slide',
                       'Rise_rise', 'Shift_slide', 'Shift_rise', 'Slide_rise', 'Twist_shift',
                       'Twist_slide',  'Twist_rise',  'Tilt_shift',  'Tilt_slide',  'Tilt_rise',
                       'Roll_shift',  'Roll_slide',  'Roll_rise',  'Slide stiffness',  'Shift stiffness',
                       'Roll stiffness', 'Rise stiffness', 'Tilt stiffness', 'Twist stiffness', 'Wedge',
                       'Direction', 'Flexibility_slide', 'Flexibility_shift', 'Entropy'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'PCPseDNC': {
        'DNA': ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'A-philicity', 'Propeller twist',
                'Duplex stability:(freeenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH', 'Electron_interaction',
                'Hartman_trans_free_energy', 'Helix-Coil_transition', 'Lisser_BZ_transition', 'Polar_interaction',
                'SantaLucia_dG', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Sugimoto_dG', 'Sugimoto_dH',
                'Sugimoto_dS', 'Duplex tability(disruptenergy)', 'Stabilising energy of Z-DNA', 'Breslauer_dS',
                'Ivanov_BA_transition', 'SantaLucia_dH', 'Stacking_energy', 'Watson-Crick_interaction',
                'Dinucleotide GC Content', 'Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'],
        'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'PCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'SCPseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                 'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
}

myDataFile = {
    'DAC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'DCC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'DACC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'TAC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'TCC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'TACC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'PseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'PCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'PCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'SCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
}
didna_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist',  'A-philicity',
              'Propeller twist', 'Duplex stability-free energy',
              'Duplex stability-disrupt energy', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
              'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
              'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
              'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
              'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
              'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt', 'Roll',
              'Shift', 'Slide', 'Rise',
              'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
              'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
              'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
              'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
              'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3', 'Propeller Twist',
              'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
              'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift', 'Slide stiffness',
              'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
              'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
              'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size', 'Twist_rise',
              'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
              'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width', 'Major Groove Depth',
              'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
              'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
              'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content', 'Slide_rise',
              'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width', 'Inclination',
              'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2', 'Twist5',
              'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide', 'Minor Groove Depth',
              'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
              'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
              'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
              'Mobility to bend towards minor groove']

tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
               'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
               'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']

dirna_list = ['Slide (RNA)', 'Adenine content', 'Hydrophilicity (RNA)', 'Tilt (RNA)', 'Stacking energy (RNA)',
              'Twist (RNA)', 'Entropy (RNA)', 'Roll (RNA)', 'Purine (AG) content', 'Hydrophilicity (RNA)1',
              'Enthalpy (RNA)1', 'GC content', 'Entropy (RNA)1', 'Rise (RNA)', 'Free energy (RNA)',
              'Keto (GT) content', 'Free energy (RNA)1', 'Enthalpy (RNA)', 'Guanine content', 'Shift (RNA)',
              'Cytosine content', 'Thymine content']
myDict = {
    'DAC': {'DNA': didna_list, 'RNA': dirna_list},
    'DCC': {'DNA': didna_list, 'RNA': dirna_list},
    'DACC': {'DNA': didna_list, 'RNA': dirna_list},
    'TAC': {'DNA': tridna_list, 'RNA': []},
    'TCC': {'DNA': tridna_list, 'RNA': []},
    'TACC': {'DNA': tridna_list, 'RNA': []},
    'PseDNC': {'DNA': didna_list, 'RNA': dirna_list},
    'PseKNC': {'DNA': didna_list, 'RNA': dirna_list},
    'PCPseDNC': {'DNA': didna_list, 'RNA': dirna_list},
    'PCPseTNC': {'DNA': tridna_list, 'RNA': []},
    'SCPseDNC': {'DNA': didna_list, 'RNA': dirna_list},
    'SCPseTNC': {'DNA': tridna_list, 'RNA': []},
}
class dynamic(dict):
    def __init__(self, d=None):
        if d is not None:
            for k,v in d.items():
                self[k] = v
        return super().__init__()

    def __key(self, key):
        return "" if key is None else key.lower()

    def __str__(self):
        import json
        return json.dumps(self)

    def __setattr__(self, key, value):
        self[self.__key(key)] = value

    def __getattr__(self, key):
        return self.get(self.__key(key))

    def __getitem__(self, key):
        return super().get(self.__key(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.__key(key), value)

pse_params = dynamic({
    'type': 'DNA',
    'index': False,
    'udi': False,
    'all_index': False,
})

def get_kmer_frequency(sequence, kmer):
    myFrequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
    return myFrequency

def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
    return CC / len(myPropertyName)

def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue)
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def make_PseKNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight, kmer):
    encodings = []
    myIndex = myDiIndex
    for sequence in fastas:
        sequence = sequence[0]
        code = []
        kmerFreauency = get_kmer_frequency(sequence, kmer)
        thetaArray = get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)

        for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
            code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
            code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))

        encodings.append(code)
    return encodings
data_path = "./dataset"
def check_Pse_arguments(args, fastas):
    # if not os.path.exists(args.file):
    #     print('Error: the input file does not exist.')
    #     sys.exit(1)
    if not 0 < args.weight <= 1:
        print('Error: the weight factor ranged from 0 ~ 1.')
        sys.exit(1)
    if not 0 < int(args.kmer) < 10:
        print('Error: the kmer value ranged from 1 - 10')
        sys.exit(1)

    fastaMinLength = 100000000
    for i in fastas:
        if len(i[0]) < fastaMinLength:
            fastaMinLength = len(i[0])
    if not 0 <= args.lamada <= (fastaMinLength - 2):
        print('Error: lamada value error, please see the manual for details.')
        sys.exit(1)

    myNum = 0
    if args.index:
        myNum = myNum + 1
    if args.udi:
        myNum = myNum + 1
    if args.all_index:
        myNum = myNum + 1

    if myNum > 1:
        print(
            'Error: argument is incorrect, "--index", "--udi" and "--all_index" can not be assigned at the same time.')
        sys.exit(1)

    myIndex = []
    myProperty = {}
    dataFile = ''

    if myNum == 0:
        myIndex = myDictDefault[args.method][args.type]
        dataFile = myDataFile[args.method][args.type]
    else:
        if args.index:
            with open(args.index) as f:
                records = f.read().strip().split('\n')
                for i in records:
                    if i in myDict[args.method][args.type]:
                        myIndex.append(i)
                    else:
                        print('Error: there is no "%s" in the index list.' % i)
                        sys.exit(1)
        if args.all_index:
            myIndex = myDict[args.method][args.type]

        if args.udi:
            if not os.path.exists(args.udi):
                print('Error: The user-defined indices file does not exist.')
                sys.exit(1)
            else:
                with open(args.udi) as f:
                    record = f.read().strip().split('\n')[1:]
                for line in record:
                    array = line.strip().split()
                    myProperty[array[0]] = array[1:]
                    myIndex.append(array[0])
        else:
            dataFile = myDataFile[args.method][args.type]

    if dataFile != '':
        if 'data' in dataFile:
            with open(data_path + '/' + dataFile, 'rb') as f:
                myProperty = pickle.load(f)
        else:
            with open(data_path + '/' + dataFile, 'r') as f:
                myProperty = readData(f)
                print(data_path + '/' + dataFile)

    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        sys.exit(1)
    return myIndex, myProperty, args.lamada, args.weight, args.kmer

def readData(file):
    dic = {}
    pattern = re.compile('[0-9]+')
    for line in file:
        a = line.split('\t')
        key = a[0]
        del a[0]
        # 判断字符串中是否函数数字
        if pattern.findall(key):
            arr = key.split(' ')
            a.insert(0, arr[len(arr)-1])
            del arr[len(arr)-1]
            key = ' '.join(arr)
        value = a
        dic[key] = value
    return dic

def get_pseknc(Seq, kmer='3', lamda='5', w='0.1'):
    pse_params.update({'kmer': kmer})
    pse_params.update({'weight': w})
    pse_params.update({'lamada': lamda})
    pse_params.update({'method': 'PseKNC'})

    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(
        pse_params, Seq)
    encodings = make_PseKNC_vector(Seq, my_property_name, my_property_value, lamada_value, weight, kmer)
    return encodings


