import os
import torch
import pytorch_lightning as pl
from transformers import OFATokenizer
import json
from torchvision import transforms
from PIL import Image
from .base_module import BaseDataModule
import numpy as np
import re
from torchvision.datasets import ImageFolder
import math



class Curriculum1ConceptDataModule(BaseDataModule):
    class Curriculum1ConceptDataset(torch.utils.data.Dataset):
        def __init__(self, config: dict, tokenizer: OFATokenizer):
            self.data_dir = config["data"]["dataset_dir"]
            self.img_dir = config["data"]["dataset_img_dir"]
            self.max_position_embeddings = config["data"]["max_position_embeddings"]
            self.tokenizer = tokenizer
            self.img_size = 224  # (224, 224), fixed; Need to modify the data if changed this resolution
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            self.code_file = config["data"]["code_file"]

            self.img2code_path = os.path.join(self.data_dir, self.code_file)
            self.img2code = self.read_json(self.img2code_path)


            self.patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((self.img_size, self.img_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            self.image_data = ImageFolder(root=self.img_dir, transform=self.patch_resize_transform)

            self.code_infilling_instruction = tokenizer.encode(
                '# Generate the object in the image\n',
                add_special_tokens=False,
                return_tensors="pt"
            ).squeeze(0)
            self.index_dict = self.build_index_dict(self.img2code)
            assert len(self.index_dict) == len(self.img2code)

        def build_index_dict(self, img2code):
            index_dict = []
            for key, value in img2code.items():
                index_dict.append(key)
            return index_dict




        def __len__(self):
            return len(self.img2code)


        def read_json(self, path):
            with open(path, 'r') as file_obj:
                json_file = json.load(file_obj)
            return json_file

        def mask_code(self, orig_code_str):
            # 'object = Cricket(loc=None, attribute=[])'
            synset_name = orig_code_str.split('object = ')[1].split('(loc=')[0]
            mask_code_str = orig_code_str.replace(synset_name, '<mask>', 1)

            decoder_tokenized_ids = self.tokenizer(
                orig_code_str, return_tensors="pt", max_length=1024, truncation=True
            ).input_ids.squeeze(0)

            return mask_code_str, decoder_tokenized_ids



        def __getitem__(self, idx):
            target_idx = self.index_dict[idx]

            code_data_str = self.img2code[target_idx]
            img, label = self.image_data[int(target_idx)]

            mask_coed_str, decoder_tokenized_ids = self.mask_code(code_data_str)

            mask_tokenized_ids = self.tokenizer(
                mask_coed_str, return_tensors="pt", add_special_tokens=False,
            ).input_ids.squeeze(0)

            new_max_position_embeddings = self.max_position_embeddings - self.code_infilling_instruction.shape[0] - 2

            mask_tokenized_ids = mask_tokenized_ids[
                                 :new_max_position_embeddings
                                 ]  # +2 for bos and eos
            # masked_ids_wo_bos_eos = mask_tokenized_ids[1:-1] if not truncated else mask_tokenized_ids[1:]

            encoder_input_ids = torch.cat(
                [
                    torch.tensor([self.tokenizer.bos_token_id]),
                    self.code_infilling_instruction.clone(),
                    mask_tokenized_ids,
                    torch.tensor([self.tokenizer.eos_token_id])
                ]
            )

            decoder_input_ids = decoder_tokenized_ids[:-1]
            decoder_target_ids = decoder_tokenized_ids[1:]

            patch_mask = torch.tensor([True])
            code_mask = torch.tensor([True])

            return {
                "metadata": None,
                "content": code_data_str,

                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
                "decoder_target_ids": decoder_target_ids,

                "patch_image": img,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
            }






    def __init__(self, config: dict, tokenizer: OFATokenizer):
        super().__init__()
        self.config = config
        self.tokenizer: OFATokenizer = tokenizer
        self.train_dataset = self.Curriculum1ConceptDataset(
            config, tokenizer
        )
