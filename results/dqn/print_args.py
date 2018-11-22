import pickle
from os import listdir

base_path = './multitask'
games = ['puddleworld']

for g in games:
    path = base_path + '/' + g
    folders = listdir(path)
    for f in folders:
        args_path = path + '/' + f + '/args.pkl'
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

