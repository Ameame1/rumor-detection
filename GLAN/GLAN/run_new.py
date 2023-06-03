import os
import pickle
import torch
from sklearn.metrics import classification_report
from model.GLAN import GLAN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def load_dataset(task):
#     X_train_tid, X_train_source, X_train_replies, y_train, word_embeddings, graph = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
#     X_dev_tid, X_dev_source, X_dev_replies, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
#     X_test_tid, X_test_source, X_test_replies, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
#     config['embedding_weights'] = word_embeddings
#     print("#nodes: ", graph.num_nodes)
#     return X_train_tid, X_train_source, X_train_replies, y_train, \
#            X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
#            X_test_tid, X_test_source, X_test_replies, y_test, graph


def load_dataset(task):
    X_train_tid, X_train_source, X_train_replies, y_train, word_embeddings, graph = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
    X_dev_tid, X_dev_source, X_dev_replies, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
    X_test_tid, X_test_source, X_test_replies, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    print("#nodes: ", graph.num_nodes)
    return X_train_tid, X_train_source, X_train_replies, y_train, \
           X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
           X_test_tid, X_test_source, X_test_replies, y_test, graph

def train_and_test(model, task):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + task + "." + model_suffix

    X_train_tid, X_train_source, X_train_replies, y_train, \
    X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
    X_test_tid, X_test_source, X_test_replies, y_test, graph = load_dataset(task)

    
    nn = model(config, graph)
    #train codes:
    # nn.fit(X_train_tid, X_train_source, X_train_replies, y_train,
    # X_dev_tid, X_dev_source, X_dev_replies, y_dev)
    # nn=nn.cuda()

    print("================================")
    nn.load_state_dict(torch.load(config['save_path']))

    #test dataset
    task_twitter15= 'twitter15'
    task_twitter16= 'twitter16'
    # task = 'twitter16'
    # task = 'weibo'

    # choose test_dataset
    if task_twitter16 == 'twitter16':
        config_16 = {
            'lr':1e-3,
            'reg':0,
            'batch_size':16,
            'nb_filters':100,
            'kernel_sizes':[3, 4, 5],
            'dropout':0.5,
            'maxlen':50,
            'epochs':30,
            'num_classes':4,
            'target_names':['NR', 'FR', 'UR', 'TR']
        }
    X_train_tid_twitter16, X_train_source_twitter16, X_train_replies_twitter16, y_train_twitter16, \
    X_dev_tid_twitter16, X_dev_source_twitter16, X_dev_replies_twitter16, y_dev_twitter16, \
    X_test_tid_twitter16, X_test_source_twitter16, X_test_replies_twitter16, y_test_twitter16, graph_twitter16 = load_dataset(task_twitter16)

    # choose test_dataset
    if task_twitter15 == 'twitter15':
        config_15 = {
            'lr':1e-3,
            'reg':0,
            'batch_size':16,
            'nb_filters':100,
            'kernel_sizes':[3, 4, 5],
            'dropout':0.5,
            'maxlen':50,
            'epochs':30,
            'num_classes':4,
            'target_names':['NR', 'FR', 'UR', 'TR']
        }
    X_train_tid_twitter15, X_train_source_twitter15, X_train_replies_twitter15, y_train_twitter15, \
    X_dev_tid_twitter15, X_dev_source_twitter15, X_dev_replies_twitter15, y_dev_twitter15, \
    X_test_tid_twitter15, X_test_source_twitter15, X_test_replies_twitter15, y_test_twitter15, graph_twitter15 = load_dataset(task_twitter15)


    task_weibo = 'weibo'
    if task_weibo == 'weibo':
        config_weibo = {
            'lr':1e-5,
            'reg':0,
            'batch_size':64,
            'nb_filters':100,
            'kernel_sizes':[3, 4, 5],
            'dropout':0.5,
            'maxlen':50,
            'epochs':30,
            'num_classes':2,
            'target_names':['NR', 'FR']
        }
    X_train_tid_weibo, X_train_source_weibo, X_train_replies_weibo, y_train_weibo, \
    X_dev_tid_weibo, X_dev_source_weibo, X_dev_replies_weibo, y_dev_weibo, \
    X_test_tid_weibo, X_test_source_weibo, X_test_replies_weibo, y_test_weibo, graph_weibo = load_dataset(task_weibo)



    
    #test dataset
    y_pred = nn.predict(X_test_tid_twitter16, X_test_source_twitter16,X_test_replies_twitter16)
    print(classification_report(y_test_twitter16, y_pred, target_names=config_16['target_names'], digits=3)) #y =Wx+b  b is the bias, models decide the W.

config = {
    'lr':1e-3,
    'reg':0,
    'batch_size':16,
    'nb_filters':100,
    'kernel_sizes':[3, 4, 5],
    'dropout':0.5,
    'maxlen':50,
    'epochs':30,
    'num_classes':4,
    'target_names':['NR', 'FR', 'UR', 'TR']
}

# choose train dataset
if __name__ == '__main__':
    #task = 'twitter16'
    # task = 'twitter16'
    task = 'twitter15'
    print("task: ", task)

    if task == 'weibo':
        config['num_classes'] = 2
        config['batch_size'] = 64
        config['reg'] = 1e-5
        config['target_names'] = ['NR', 'FR']

    model = GLAN
    train_and_test(model, task)


   

