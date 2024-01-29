import torch
import importlib
import io
import os
import numpy as np
import data_config
import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, rand_score
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import argparse
import warnings
warnings.filterwarnings("ignore")

def myloads(jstr):
    return yaml.safe_load(io.StringIO(jstr))


parser = argparse.ArgumentParser(description='model')
parser.add_argument('-f', dest='argFile', type=str, required=False,
                    default=None,
                    help='Specify the test parameter file via the YAML file.')
# start_time= str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
parser.add_argument('-log', dest='log_path', type=str, required=False,
                    # default='',
                    help='Specify the path where the results are stored')
parser.add_argument('-m', dest='metrics', type=str, required=False,
                    # default="",
                    help='Verify the list of metrics, split by commas')
parser.add_argument('-a', dest='alg', type=str, required=False,
                    # default="",
                    help='algorithm')
parser.add_argument('-d', dest='data', type=str, required=False,
                    # default="data",
                    help='Data Directory')
parser.add_argument('-r', dest='params', type=myloads, required=False,
                    # default="{}",
                    help='''Algorithm parameters in JSON format, such as "{d:20,lr:0.1,n_itr:1000}" ''')
import os
import zipfile

import numpy as np
import pandas as pd
class MetricAbstract:
    def __init__(self):
        self.bigger= True
    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")
class RI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return rand_score(groundtruth, pred)


class NMI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return normalized_mutual_info_score(groundtruth, pred)
def list_file_with_prefix(paths,prefix):
    result=[]
    for data_file in paths:
        s=data_file.split('/')[1]
        if s.startswith(prefix):
            result.append(data_file)
    return result

def parse_data(data_dir):
    if os.path.isfile(data_dir):
        z = zipfile.ZipFile(data_dir, mode='r')
        dir_list = z.namelist()
        path_train = list_file_with_prefix(dir_list, "TRAIN")
        path_test = list_file_with_prefix(dir_list, "TEST")
    else:
        print('data_dir should  be a zip file !')
    train_set = csv_to_X_y(path_train,z)
    test_set = csv_to_X_y(path_test,z)

    # Combine training set data and test set data
    X = np.concatenate((train_set[0], test_set[0]), axis=0)
    y = np.concatenate((train_set[1], test_set[1]), axis=0)
    return (X, y)


def csv_to_X_y(filepath,z):
    list_X = []
    y = None
    for path in filepath:
        dataframe = pd.read_csv(z.open(path), header=None)
        if path.endswith('label.csv'):
            y = np.squeeze(dataframe.values)
        else:
            list_X.append(np.expand_dims(dataframe.values, axis=-1))
    X = np.concatenate(list_X, axis=-1)

    assert (y is not None)
    assert (y.shape[0] == X.shape[0])

    X = np.transpose(X, (0, 2, 1))
    return X, y
def set_seed(seed=2333):
    import random,os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
def update_parameters(param:dict, to_update:dict)->dict:
    for k ,v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k],(dict,)):
                    param[k].update(to_update[k])
                else:
                    param[k]=to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param
def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls
def create_obj_from_json(js):
    if isinstance(js,dict):
        assert len(js.items()) == 1
        for key,values in js.items():
            cls = my_import(key)()
            if isinstance(values,dict):
                for k, v in values.items():
                    setattr(cls, k, create_obj_from_json(v))
            elif values is None:
                pass
            else:
                raise Exception("Not valid parameters(Must be dict):",values)
            return cls
    elif isinstance(js,(set,list)):
        return [create_obj_from_json(x) for x in js]
    else:
        return js
def main():

    #Read the model parameters from the configuration file
    args = parser.parse_args()
    if args.argFile is not None:
        with open(args.argFile) as infile:
            filedict = yaml.safe_load(infile)
    else:
        filedict = {}

    # Read the data set path
    data_dir = filedict['data_dir']
    data_dir_list = os.listdir(data_dir)
    data_dir_list.sort()
    # Traverse the data set
    for data in data_dir_list:
        set_seed()
        file_path = os.listdir(os.path.join(data_dir, data))[0]
        args.data = data_dir + '/' + data + '/' + file_path
        arg_dict = {
            "algorithm": args.alg,
            "algorithm_parameters": args.params,
            "data_dir": args.data,
            "log_path": args.log_path}
        #Update the parameters in the yaml configuration file to
        update_parameters(filedict, arg_dict)
        algorithm = create_obj_from_json({filedict['algorithm']: filedict['algorithm_parameters']})

        model_save = './model_save/' + data
        if not os.path.exists(model_save):
            os.makedirs(model_save)
        algorithm.model_save_path = model_save+'/'+data+'.pt'
        # The output dimension of the convolutional layer
        algorithm.CNNoutput_channel = data_config.CNNoutput_channel[data]

        #loading the data
        data_dir = filedict['data_dir']
        ds = parse_data(data_dir)
        # Evaluation index
        metrics = [NMI(),RI()]
        algorithm.train(ds, valid_ds=None, valid_func=metrics)
        pred = algorithm.predict(ds)
        true_label = np.array(ds[1])
        results = [m(true_label, pred) for m in metrics]
        metrics_name = [str(m) for m in metrics]
        print("RESULTS="+str(dict(zip(metrics_name, results))))
if __name__ == '__main__':
    main()
