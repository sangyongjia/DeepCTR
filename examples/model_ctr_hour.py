import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from tf.keras.utils import multi_gpu_model
# from deepctr.models import FM, DeepFM, WDL, PNN, FNN, DCN, AFM, xDeepFM, AutoInt, ONN, FiBiNET, FLEN

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names

from deepctr.models import DeepFM

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
# tf.logging.set_verbosity(tf.logging.INFO)
import yaml

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

if __name__ == "__main__":
    print("start")


    def load_conf(filename):
        with open("../conf/" + filename, 'r') as f:
            return yaml.load(f)


    conf = load_conf("model_ctr_hour.yaml")

    all_columns = conf["columns_name_in_data_file"]
    columns_name_index = {}
    for i in range(len(all_columns)):
        columns_name_index[all_columns[i]] = i
    # print("column_names_index_dict:",column_names_index_dict)
    vocabulary_size_val = 1
    select_columns_name = []
    vocabulary_size = conf['vocabulary']
    use_weight = True
    use_hour_features = True
    varlen_feature_columns = []
    if use_weight:
        if use_hour_features:
            for feat_name in all_columns:
                if feat_name[-6:] == 'weight' or feat_name in ['ctr_label', 'cvr_label']:
                    select_columns_name.append(feat_name)
                    continue
                for key in vocabulary_size.keys():
                    if key in feat_name:
                        vocabulary_size_val = vocabulary_size[key]
                        print("key:{0},size:{1}".format(key, vocabulary_size_val))
                        break
                print("size:{0}".format(vocabulary_size_val))
                varlen_feature_columns.append(VarLenSparseFeat(
                    SparseFeat(feat_name, vocabulary_size=vocabulary_size_val + 1, embedding_dim=4, use_hash=False), maxlen=1,
                    combiner='mean', weight_name=feat_name + '_weight'))
                select_columns_name.append(feat_name)
        else:
            for feat_name in all_columns:
                if feat_name[:4] == 'hour':
                    continue
                if feat_name[-6:] == 'weight':
                    select_columns_name.append(feat_name)
                    continue
                for key in vocabulary_size.keys():
                    if key in feat_name:
                        vocabulary_size_val = vocabulary_size[key]
                        break
                varlen_feature_columns.append(VarLenSparseFeat(
                    SparseFeat(feat_name, vocabulary_size=vocabulary_size_val + 1, embedding_dim=4), maxlen=1,
                    combiner='mean', weight_name=feat_name + '_weight'))
                select_columns_name.append(feat_name)
    else:
        if use_hour_features:
            for feat_name in all_columns:
                if feat_name[-6:] == 'weight':
                    continue
                for key in vocabulary_size.keys():
                    if key in feat_name:
                        vocabulary_size_val = vocabulary_size[key]
                        break
                varlen_feature_columns.append(VarLenSparseFeat(
                    SparseFeat(feat_name, vocabulary_size=vocabulary_size_val + 1, embedding_dim=4), maxlen=1,
                    combiner='mean', weight_name=None))
                select_columns_name.append(feat_name)
        else:
            for feat_name in all_columns:
                if feat_name[:4] == 'hour' or feat_name[-6:] == 'weight':
                    continue
                for key in vocabulary_size.keys():
                    if key in feat_name:
                        vocabulary_size_val = vocabulary_size[key]
                        break
                varlen_feature_columns.append(VarLenSparseFeat(
                    SparseFeat(feat_name, vocabulary_size=vocabulary_size_val + 1, embedding_dim=4), maxlen=1,
                    combiner='mean', weight_name=None))
                select_columns_name.append(feat_name)

    select_columns_name.remove('ctr_label')

    select_columns_index_name_tuple = []
    for name in select_columns_name:
        select_columns_index_name_tuple.append((columns_name_index[name], name))
    # print("select_columns_index_name_tuple:",select_columns_index_name_tuple)
    select_columns_index_name_tuple.sort(key=lambda k: k[0])
    # print("select_columns_index_name_tuple:\n", select_columns_index_name_tuple)

    select_columns_index = [value[0] for value in select_columns_index_name_tuple]
    select_columns_name1 = [value[1] for value in select_columns_index_name_tuple]
    print("\nin model select_columns_name:\n", select_columns_name1)
    print("\nin model select_columns_index:\n", select_columns_index)
    # print("select_columns_index len:",len(select_columns_index))
    vocabulary_size = conf["vocabulary"]

    # 将来新增列的时候这个位置是个需要调整的点
    # defaults = [[0], [tf.constant(0, dtype=tf.int64)], [tf.constant(0, dtype=tf.int64)]] + [[0.0]] * (
    #            len(select_columns_name) - 3)
    defaults = [[0.0]] * len(select_columns_name1)
    # print("\ndefaults:\n\n",len(defaults))
    select_columns_name1.remove("cvr_label")
    train_data_path = conf['train_data_path']
    eval_data_path = conf['eval_data_path']


    def get_dataset(file_path=train_data_path, perform_shuffle=True, repeat_count=1, batch_size=1024):
        def decode_csv(line):
            res = dict()
            parsed_line = tf.io.decode_csv(line, record_defaults=defaults, field_delim=',',
                                           select_cols=select_columns_index)
            label = parsed_line[0]
            print("\nlen\n", len(parsed_line))
            print("\nparsed_line:\n", parsed_line)
            del parsed_line[0]
            features = parsed_line  # Everything but last elements are the features
            for i, name in enumerate(select_columns_name1):
                if name[-6:] == 'weight':
                    res[name] = [features[i]]
                else:
                    res[name] = features[i]
                pass

            # d = dict(zip(select_columns_name1, features)), label
            d = res, label
            return d

        dataset = (tf.data.TextLineDataset(file_path, buffer_size=20000000, num_parallel_reads=20)  # Read text file
                   .map(decode_csv, num_parallel_calls=20))  # Trans:form each elem by applying decode_csv fn
        if perform_shuffle:
            # Randomizes input using a window of 256 elements (read into memory)
            dataset = dataset.shuffle(buffer_size=batch_size * 8)
        dataset = dataset.batch(batch_size=batch_size)  # Batch size to use
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(batch_size * 2)
        dataset = dataset.cache()
        dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
        return dataset


    linear_feature_columns = varlen_feature_columns
    dnn_feature_columns = varlen_feature_columns
    callbacks = []
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1','/gpu:2','/gpu:3'])
    # strategy = tf.distribute.MirroredStrategy(devices=['/gpu:3'])
    with strategy.scope():
        model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024, 512, 256], task='binary',
                       dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False)
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
    # model.run_eagerly = True
    model.fit_generator(generator=get_dataset(), steps_per_epoch=None, epochs=10, verbose=2, callbacks=callbacks,
                        validation_data=get_dataset(eval_data_path), validation_steps=None, validation_freq=1,
                        class_weight=None,
                        max_queue_size=100, workers=10, use_multiprocessing=False, shuffle=True, initial_epoch=0)
