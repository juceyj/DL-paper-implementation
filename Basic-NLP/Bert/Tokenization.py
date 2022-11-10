'''
Author: Jiayi Liu
Date: 2022-11-10 17:34:02
LastEditors: Jiayi Liu
LastEditTime: 2022-11-10 19:58:05
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
from transformers import BertTokenizer
text = 'I like natural language progressing!'
BertToken = BertTokenizer.from_pretrained('bert-base-uncased')
subword = BertToken.tokenize(text)
token2id = BertToken.convert_tokens_to_ids(subword)
print(BertToken(text))
print(token2id)


# text = '[CLS] 武1松1打11老虎 [SEP] 你在哪 [SEP]'
# tokenized_text = tokenizer.tokenize(text)#切词 方式1
# token_samples_a = tokenizer.convert_tokens_to_ids(tokenized_text)#只返回token_ids,手动添加CLS与SEP

# token_samples_b=tokenizer(text)#返回一个字典，包含id,type,mask，无须手动添加CLS与SEP 方式2

# token_samples_c=tokenizer.encode(text=text)#只返回token_ids，无须手动添加CLS与SEP 方式3

# token_samples_d=tokenizer.encode_plus(text=text,max_length=30,return_tensors='pt')#方式4 返回一个字典，包含id,type,mask，无须手动添加CLS与SEP，可以指定返回类型与长度


