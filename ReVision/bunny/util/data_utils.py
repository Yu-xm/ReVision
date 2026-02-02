import os
import copy
import pickle
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional
import torch
import transformers
from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset
from bunny import conversation as conversation_lib
from bunny.util.mm_utils import tokenizer_image_token

import random


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_aspect_ratio: str = field(default=None)


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            # print('<----------------------------------------------->')
            # print('for111')
            # print('<----------------------------------------------->')
            if rou == "":
                # print('<----------------------------------------------->')
                # print('for222')
                # print('<----------------------------------------------->')
                break
            
            # print('<----------------------------------------------->')
            # print('for444')
            # print('<----------------------------------------------->')

            parts = rou.split(sep)
            if len(parts) != 2:
                # print('<----------------------------------------------->')
                # print('for333')
                # print('<----------------------------------------------->')
                break
            parts[0] += sep

            # print('<----------------------------------------------->')
            # print('for555')
            # print('<----------------------------------------------->')

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_bunny_with_bos(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        end_token_cnt = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):

            # print('<----------------------------------------------->')
            # print('111')
            # print('<----------------------------------------------->')

            if rou == "":
                # print('<----------------------------------------------->')
                # print('222')
                # print('<----------------------------------------------->')
                break
            
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            end_token_cnt += 1
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        # print('00000000000')
        return preprocess_plain(sources, tokenizer)

    if conversation_lib.default_conversation.version == "bunny":
        # print('11111111111')
        return preprocess_bunny(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version in {"minicpm", "llama"}:
        # print('22222222222')
        return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)
    # temporarily fix
    # Phi-3 June 2024 Update changes bos_token behavior
    elif conversation_lib.default_conversation.version == "phi3":
        if len(tokenizer('').input_ids) == 0:
            # print('33333333333')
            return preprocess_bunny(sources, tokenizer, has_image=has_image)
        else:
            # print('44444444444')
            return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        # 【修改点2】：预先过滤掉没有 caption 的数据，防止 __getitem__ 返回 None 导致崩溃
        self.list_data_dict = [x for x in list_data_dict if 'caption' in x]
        print(f"Loaded {len(list_data_dict)} raw samples, filtered to {len(self.list_data_dict)} valid samples (with captions).")

	# <----------------------------------------------------->
	# bunny	
	# <----------------------------------------------------->	

        # ours
        # folder_path = '/llm-align/liuchonghan/embed_0113/exp_embedding/embedding_bunny/text_ours/ours'
	
	# trace
        #folder_path = '/mnt/public/data/h200/victor/xiaomin/datasets/bunny_densefusion_image_pkl/exp_embedding/embedding_bunny/image'
        # folder_path = '/mnt/public/data/h200/victor/xiaomin/datasets/'
        folder_path = '/mnt/public/data/h200/victor/xiaomin/datasets/internvl_dataset/embedding'
        # <----------------------------------------------------->
	# densefusion	
	# <----------------------------------------------------->

        # ours
        # folder_path = '/llm-align/liuchonghan/embed_0113/exp_embedding/embedding_densefusion/text_ours/ours'

	# trace
        # folder_path = '/llm-align/liuchonghan/embed_0113/exp_embedding/embedding_densefusion/text_trace/trace'

        print('<----------------------------------------------------->')
        print(folder_path)
        print('<----------------------------------------------------->')
        
        # 【修改点1】：使用 Dictionary (Hash Map) 代替 List，实现 O(1) 查找
        self.embeddings = {}

        if os.path.exists(folder_path):
            print(f"Loading embeddings from {folder_path}...")
            for filename in os.listdir(folder_path):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            batch_data = pickle.load(f)
                            # 建立 id -> embed 的映射
                            for item in batch_data:
                                self.embeddings[str(item.get('id'))] = item.get('embed')
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
            print(f"Loaded {len(self.embeddings)} embeddings.")
        else:
            print(f"ERROR: Folder path {folder_path} does not exist.")


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        
        # sources
        # {
        #     "id": "0010278167",
        #     "caption": ***,  
        #     "conversations": [
        #         {
        #             "from": "human", 
        #             "value": "<image>\nWhat is this?"
        #         }, 
        #         {
        #             "from": "gpt", 
        #             "value": "***"
        #         }
        #     ]
        # }

        # caption 有两个作用：
        # 1. 替代 image
        # 2. 作为 pretrain 阶段 imaginary image 的 caption 训练数据
        sources = self.list_data_dict[i]

        # print('<------------------------------------->')
        # print('sources1: \n')
        # print(sources)
        # print('<------------------------------------->')

        # 【修改点2】：由于在 __init__ 已经过滤，这里一定有 caption，无需 else return None
        # 作用 2：作为 pretrain 阶段 imaginary image 的 caption 训练数据 TODO
        sources_processed = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in [sources]]), self.data_args)

        # print('<------------------------------------->')
        # print('sources2: \n')
        # print(sources)
        # print('<------------------------------------->')
        
        # 由于数据格式一致，所以无需修改 preprocess
        data_dict = preprocess(
            sources_processed, # 注意这里使用处理后的 sources_processed
            self.tokenizer,
            has_image=True # 既然经过过滤，则认为包含图像信息
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])


        # print('<------------------------------------->')
        # print('self.list_data_dict[i]: \n')
        # print(self.list_data_dict[i])
        # print('<------------------------------------->')
        
        # 【修改点1】：使用字典 O(1) 查找 embedding
        sid = str(sources.get('id'))
        if sid in self.embeddings:
            # print('IN THERE!')
            data_dict['embed'] = self.embeddings[sid]
        
        # print('<------------------------------------->')
        # print('data_dict2: \n')
        # print(data_dict)
        # print('<------------------------------------->')

        # dict:
        # {
        #     input_ids,
        #     labels,
        #     embed
        # }

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        # print('<instances---------------------------------------------------->')
        # print(instances)
        # print('<---------------------------------------------------->')

        if 'embed' in instances[0]:

            embeds = [instance['embed'] for instance in instances]
            embeds_tensor = torch.stack([torch.from_numpy(embed) for embed in embeds])

            # print('<In data utils-------------------------------------------->')
            # print(embeds_tensor)
            # print('<In data utils-------------------------------------------->')

            # 可选：将数据类型转换为浮点数（如果尚未）
            # embeds_tensor = embeds_tensor.float()
            batch['embeds'] = embeds_tensor

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
