# import word2vec
import fasttext
import sys
import os

'''
Usage: python get_vocab.py /path/to/vocab.bin
'''
w2v_file = "/home/ubuntu/filestore/keshav/im2recipe/data"
# model = word2vec.load(w2v_file)
model = fasttext.load_model(w2v_file + "/im2recipe_myntra_vocab.bin")
vocab =  model.words

print("Writing to %s..." % os.path.join(os.path.dirname(w2v_file),'vocab.txt'))
f = open(os.path.join(os.path.dirname(w2v_file),'im2recipe_myntra_vocab.txt'),'w')
f.write("\n".join(vocab))
f.close()
