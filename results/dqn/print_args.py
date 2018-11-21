import pickle
from os import listdir

base_path = './multitask'
folders = listdir(base_path)

for f in folders:
    args_path = base_path + '/' + f + '/args.pkl'
    with open(args_path, 'rb') as input_file:
        args = pickle.load(input_file)
        if args.multi:
            if args.reg_type is None:
                print(f, '-> multi', args.features)
            else:
                print(f, '-> multi', args.features, args.reg_type,
                      args.reg_coeff, args.k)
        else:
            reg = args.reg_type if args.reg_type is not None else ''
            print(f, '-> single', args.features, reg)

