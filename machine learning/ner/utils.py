# -*- coding: utf-8 -*-

__author__ = 'iamyaolan@163.com'
import os
import codecs
import pickle
import numpy as np
import tensorflow as tf

def uniform_tensor(shape,name,dtype='float32'):
    '''
    initialize tensor
    args:
        shape: tuple
        name: str
    returns:
        tensor
    '''
    return tf.random_uniform(shape=shape,minval=-1.0,maxval=1.0,dtype=tf.float32,name=name)

def get_sequence_actual_length(tensor):
    '''
    get the actual length of tensor
    args:
        tensor: a 2d tensor with shape (batch_size,max_len)
    return:
        actual_len: a vector with length [batch_size]
    '''
    actual_length = tf.reduce_sum(tf.sign(tensor),axis=1)
    return tf.cast(actual_length,tf.int32)

def zero_nil_slot(t,name=None):
    '''
    overwrite the nil_slot (first 1 rows) of the input tensor with zeros.
    args:
        t: 2D tensor
        name： str
    returns:
        same shape as t
    '''
    with tf.name_scope('zero_nil_slot'):
        s = tf.shape(t)[1]
        z = tf.zeros([1,s],dtype=tf.float32)
        return tf.concat(axis=0,name=name,
                         values=[z,tf.slice(t,[1,0],[-1,-1])])

def shuffle_matrix(*args,**kw):
    '''
    shuffle 句矩阵
    '''
    seed = kw['seed'] if 'seed' in kw else 1337
    for arg in args:
        np.random.seed(seed)
        np.random.shuffle(arg)

def create_dictionary(token_dict,dic_path,start=0,sort=False,min_count=None,lower=False,overwrite=False):
    '''
    construct dict, then write this dict to pkl files.
    :param token_dict: dict, [token_1:count_1, token_2:count_2,...]
    :param dic_path: 需要保存的路径（以pkl结尾）
    :param start: int, voc起始下标，默认为0
    :param sort: 是否按照频率排序，若为False，则按items排序
    :param min_count: 词最少出现次数，低于此值的词被过滤
    :param lower: 是否转为小写
    :param overwrite: 是否覆盖之前的文件
    :return:
        voc size: int
    '''
    if os.path.exists(dic_path) and not overwrite:
        return 0
    voc = dict()
    if sort:
        #sort
        token_list = sorted(token_dict.items(),key=lambda d: d[1],reverse=True)
        for i,item in enumerate(token_list):
            if min_count and item[1] < min_count:
                continue
            index = i + start
            key = item[0]
            voc[key] = index
    else:
        # 按items排序
        if min_count:
            items = sorted([item[0] for item in token_dict.items() if item[1] >= min_count])
        else:
            items = sorted([item[0] for item in token_dict.items()])
        for i,items in enumerate(items):
            item = item if not lower else item.lower()
            index = i + start
            voc[item] = index
    #写入文件
    file = open(dic_path,'wb')
    pickle.dump(voc,file)
    file.close()
    return len(voc.keys())

def map_item2id(items,voc,max_len,none_word=1,lower=False,init_value=0,allow_error=True):
    '''
    将word/pos等映射为id
    :param items: list,待映射列表
    :param voc: 词表
    :param max_len: int,序列最大成都
    :param none_word: 未登录词标号，默认为0
    :param lower: 是否转换为小写
    :param init_value: 初始化的值，默认未0
    :param allow_error:
    :return:
        arr: np.array, dtype=int32, shape=[max_len,]
    '''
    assert type(none_word) == int
    arr = np.zeros((max_len,),dtype='int32') + init_value
    min_range = min(max_len,len(items))
    for i in range(min_range): # 若items长度大于max_len，则被截断
        item = items[i] if not lower else items[i].lower()
        if allow_error:
            arr[i] = voc[item] if item in voc else none_word
        else:
            arr[i] = voc[item]
    return arr

def build_lookup_table(vec_dim,token2id_dict,token2vec_dict=None, token_voc_start=1):
    '''
    构建look-up table
    :param vec_dim: 向量维度
    :param token2id_dict: 键： token，值： id
    :param token2vec_dict: key: token, value: np.array(预训练的词向量)
    :param token_voc_start: int, 起始位置
    :return:
        token_weight: np.array, shape=(table_size,dim)
        unknow_token_count: int,未登录词数量
    '''
    unknow_token_count = 0
    token_voc_size = len(token2id_dict.keys()) + token_voc_start

    if token2vec_dict is None: # randomly initialize
        token_weight = np.random.normal(size=(token_voc_size,vec_dim)).astype('float32')
        for i in range(token_voc_start):
            token_weight[i,:] = 0.
        return  token_weight,0

    token_weight = np.zeros((token_voc_size,vec_dim),dtype='float32')
    for token in token2id_dict:
        index = token2id_dict[token]
        if token in token2vec_dict:
            token_weight[index,:] = token2vec_dict[token]
        else:
            unknow_token_count += 1
            random_vec = np.random.uniform(-0.25,0.25,size=(vec_dim,)).astype('float32')
            token_weight[index,:] = random_vec
    return  token_weight,unknow_token_count

def embedding_txt2pkl(path_txt,path_pkl):
    '''
    将txt文件转换为pkl文件
    :param path_txt: str, txt embedding file path
    :param path_pkl: pkl file path
    :return:
        none
    '''
    print('convert txt to pkl...')
    from gensim.models.keyedvectors import KeyedVectors
    assert path_txt.endswith('txt')
    word_vectors = KeyedVectors.load_word2vec_format(path_txt,binary=False)
    word_dict = {}
    for word in word_vectors.vocab:
        word_dict[word] = word_vectors[word]
    with open(path_pkl,'wb') as file_w:
        pickle.dump(word_dict,file_w)
    print('.txt file has wrote to: %s!\n - embedding dim is %d.' %
          (path_pkl, word_vectors.vector_size))


def load_embed_from_txt(path):
    '''
    读取txt文件格式的embedding
    :param path: str, 路径
    :return:
        embed_dict: dict
    '''
    file_r = codecs.open(path,'r',encoding='utf-8')
    line = file_r.readline()
    voc_size,vec_dim = map(int,line.split(' '))
    embedding = dict()
    line = file_r.readline()
    while line:
        items = line.split(' ')
        item = items[0]
        vec = np.array(items[1:],dtype='float32')
        embedding[item] = vec
        line = file_r.readline()
    return embedding,vec_dim