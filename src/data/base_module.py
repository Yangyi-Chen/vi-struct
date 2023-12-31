import torch
import pytorch_lightning as pl

class BaseDataModule(pl.LightningDataModule):
    def collate_fn(self, batch):
        """Folloing OFA implementation:
        https://github.com/OFA-Sys/OFA/blob/3b181d74d5fd1a5bc3e804813b5f2515633740d8/data/pretrain_data/unify_dataset.py#L55
        """
        metadata = [d["metadata"] for d in batch]
        content = [d["content"] for d in batch]


        # pad to the same length for batched training
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["encoder_input_ids"] for d in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["decoder_input_ids"] for d in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        decoder_target_ids = torch.nn.utils.rnn.pad_sequence(
            [d["decoder_target_ids"] for d in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        assert decoder_input_ids.shape == decoder_target_ids.shape, \
            "decoder_input_ids and decoder_target_ids should have the same shape"
        
        # length statistics
        src_lengths = torch.LongTensor([
            d["encoder_input_ids"].ne(self.tokenizer.pad_token_id).long().sum()
            for d in batch
        ])
        tgt_lengths = torch.LongTensor([
            d["decoder_input_ids"].ne(self.tokenizer.pad_token_id).long().sum()
            for d in batch
        ])
        num_tokens = tgt_lengths.sum().item()

        # patch and mask
        if isinstance(batch[0]["patch_image"], torch.Tensor):
            patch_images = torch.stack([d["patch_image"] for d in batch])
        else:
            assert batch[0]["patch_image"] == None
            patch_images = None
        
        if isinstance(batch[0]["patch_mask"], torch.Tensor):
            patch_masks = torch.stack([d["patch_mask"] for d in batch])
        else:
            assert batch[0]["patch_mask"] == None
            patch_masks = None

        if isinstance(batch[0]["code_mask"], torch.Tensor):
            code_masks = torch.stack([d["code_mask"] for d in batch])
        else:
            assert batch[0]["code_mask"] == None
            code_masks = None

        return {
            "metadata": metadata,
            "content": content,

            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_target_ids": decoder_target_ids,

            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "code_masks": code_masks,

            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "num_tokens": num_tokens,
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
