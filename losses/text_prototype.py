import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer

def select_representative(embedding_matrix, k):
    m, n = embedding_matrix.shape
    result = np.empty(m, dtype=int)
    cores = embedding_matrix[np.random.choice(np.arange(m), k, replace=False)]
    while True:
        d = np.square(np.repeat(embedding_matrix, k, axis=0).reshape(m, k, n) - cores)
        distance = np.sqrt(np.sum(d, axis=2))
        index_min = np.argmin(distance, axis=1)
        if (index_min == result).all():
            return cores

        print('[{}/{}]'.format(sum(index_min == result),m))
        result[:] = index_min
        for i in range(k):
            items = embedding_matrix[result == i]
            cores[i] = np.mean(items, axis=0)
    return cores

def select_prototype(model_dir='../models/gpt2',prototype_dir='../losses',provide='representative',number_of_prototype=10):
    model_path = os.path.join(model_dir, 'pytorch_model.bin')
    state_dict = torch.load(model_path)
    embedding_matrix = state_dict['wte.weight']
    if isinstance(provide,list):
        print('----Select provided text prototypes----')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokens = tokenizer.tokenize(''.join(provide))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        embeddings = embedding_matrix[token_ids]
        torch.save(embeddings, os.path.join(prototype_dir, 'text_prototype_provided.pt'))

    elif provide=='random':
        print('----Select {} random text prototypes----'.format(number_of_prototype))
        token_ids=[random.randint(0,len(embedding_matrix)) for i in range(number_of_prototype)]
        embeddings = embedding_matrix[token_ids]
        torch.save(embeddings, os.path.join(prototype_dir, 'text_prototype_random.pt'))

    else:
        print('----Select {} representative text prototypes----'.format(number_of_prototype))
        embedding_matrix = embedding_matrix.numpy()
        embeddings=select_representative(embedding_matrix,number_of_prototype)
        embeddings=torch.from_numpy(embeddings)
        torch.save(embeddings, os.path.join(prototype_dir, 'text_prototype_representative.pt'))
    return embeddings,embeddings.size()


if __name__ == '__main__':

    embeddings,size=select_prototype()
    print(size)

