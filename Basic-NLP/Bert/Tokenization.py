'''
Author: Jiayi Liu
Date: 2022-11-10 17:34:02
LastEditors: Jiayi Liu
LastEditTime: 2022-11-10 20:19:09
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
from transformers import BertTokenizer
text = 'I like natural language progressing!'
BertToken = BertTokenizer.from_pretrained('bert-base-uncased')
subword = BertToken.tokenize(text)