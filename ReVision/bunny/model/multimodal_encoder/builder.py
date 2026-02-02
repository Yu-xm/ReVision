import torch
import torch.nn as nn

import os
import pickle


# class LLM2CLIPTextTower(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.image_processor = CLIPImageProcessor.from_pretrained("/data/xmyu/Bunny_all/checkpoints/clip-vit-large-patch14-336")
#         self.model = AutoModel.from_pretrained(
#             "/data/xmyu/Bunny_all/checkpoints/LLM2CLIP-Openai-L-14-336", 
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True).to('cuda').eval()
        
#         self.model.requires_grad_(False)
        

#         self.llm_model_name = '/data/xmyu/Bunny_all/checkpoints/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
#         config = AutoConfig.from_pretrained(
#             self.llm_model_name, trust_remote_code=True
#         )
#         self.llm_model = AutoModel.from_pretrained(self.llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
#         self.llm2clip_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
#         self.llm_model.config._name_or_path = '/data/xmyu/Bunny_all/checkpoints/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
#         self.l2v = LLM2Vec(self.llm_model, self.llm2clip_tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

#         self.is_loaded = True

#     def forward_img(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
#                                                       output_hidden_states=True)
#                 image_features.append(image_feature)
#         else:
#             image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
#                                                    output_hidden_states=True)

#         return image_features
    
#     def forward(self, captions):

#         if type(captions) is list:
#             caption_features = self.l2v.encode(captions, convert_to_tensor=True).to('cuda')

#         with torch.no_grad(), torch.cuda.amp.autocast():

#             caption_features = self.model.get_text_features(caption_features)
#             caption_features /= caption_features.norm(dim=-1, keepdim=True)

#         print('<-------------------------->')
#         print(caption_features.shape)
#         print('<-------------------------->')
        
#         return caption_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def hidden_size(self):
#         return 1280

# def build_vision_tower(vision_tower_cfg, **kwargs):

#     return LLM2CLIPTextTower()


class LLM2CLIPTextTower(nn.Module):
    def __init__(self):
        super().__init__()

        folder_path = '/data/xmyu/Bunny_all/data/embeddings/pkl/captions'  # 替换为实际的文件夹路径
        self.embeddings = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.pkl'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'rb') as f:
                    self.embeddings.extend(pickle.load(f))
    
    def forward(self, ids):
        embeds = []
        for id_ in ids:
            embed = next((d['embed'] for d in self.embeddings if d['id'] == id_), None)
            if embed is None:
                raise KeyError(f"ID {id_} not found in embeddings.")
            embeds.append(embed)
        return embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1280, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    @property
    def hidden_size(self):
        return 1280

def build_vision_tower(vision_tower_cfg, **kwargs):

    return LLM2CLIPTextTower()