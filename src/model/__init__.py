import torch
import pytorch_lightning as pl
from transformers import OFATokenizer, OFAModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from src.utils import add_space_to_tokenizer



ignore_index_li = [1, 40412, 5457, 1640, 26516, 5214, 29802, 6, 21643, 49310, 45587, 2, 49310, 7479, 10936, 26907, 43, 10431, 15745, 877, 42640, 50118, 8136, 1258, 7081]
number_li = [288, 134, 176, 246, 306, 245, 401, 406, 398, 466, 698, 1225, 1092, 1558, 1570, 996, 1549, 1360, 1366, 1646, 844, 2146, 2036, 1922, 1978, 1244, 2481, 2518, 2517, 2890, 541, 2983, 2881, 3103, 3079, 2022, 3367, 3272, 3170, 3416, 1749, 4006, 3714, 3897, 3305, 1898, 3761, 3706, 3818, 3414, 1096, 4708, 4429, 4540, 4283, 3118, 4419, 4390, 4432, 4156, 2466, 5606, 5379, 5449, 4027, 3506, 4280, 4111, 4671, 4563, 3083, 5339, 4956, 5352, 5243, 2545, 5067, 4718, 5479, 5220, 2940, 6668, 6551, 6361, 6232, 4531, 5334, 5677, 4652, 5046, 3248, 6468, 6617, 6478, 6405, 4015, 5607, 6750, 5208, 2831]
ignore_index_li_with_number = ignore_index_li + number_li


class OFAModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Build tokenizer
        self.tokenizer = OFATokenizer.from_pretrained(
            config["model"]["ckpt_dir"]
        )
        _prevlen = len(self.tokenizer)
        self.tokenizer = add_space_to_tokenizer(self.tokenizer)
        print(f"Added space to tokenizer for codegen, new length: {_prevlen} -> {len(self.tokenizer)}")
        # NOTE: since we added these special spaces,
        # we need to use spaces_between_special_tokens=True when decoding, otherwise, we will get extra spaces between these
        # special "spaces" and the tokens around them.
        # https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/tokenization_utils.py#L951

        # 2. Build model
        self.model = OFAModel.from_pretrained(config["model"]["ckpt_dir"])
        # resize token embeddings
        # this will extend both self.model.encoder.embed_tokens and self.model.decoder.embed_tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # this will rebuild self.model.decoder.output_projection.weight
        # and use embed_tokens.weight as the initial value
        assert self.model.config.share_decoder_input_output_embed
        self.model.decoder.build_output_projection(self.model.config)

        self.save_hyperparameters()

    def forward(self, batch):
        encoder_input_ids = batch["encoder_input_ids"]
        decoder_input_ids = batch["decoder_input_ids"]
        patch_images = batch["patch_images"]
        patch_masks = batch["patch_masks"]

        # non-pad positions are 1, pad positions are 0
        decoder_attention_mask = decoder_input_ids.ne(
            self.tokenizer.pad_token_id
        )
        # Forward
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=encoder_input_ids,
            patch_images=patch_images,
            patch_images_2=None,
            patch_masks=patch_masks,
            decoder_input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask
        )
        logits = outputs.logits
        ret = {
            "outputs": outputs,
        }
        # Calculate loss if decoder_target_ids is provided
        if "decoder_target_ids" in batch:
            convert_logits = logits.view(-1, logits.size(-1))

            # replace pad_token_id with -100 to ignore loss in decoder_target_ids
            decoder_target_ids = batch["decoder_target_ids"]
            convert_decoder_target_ids = decoder_target_ids.view(-1)
            for i in range(convert_decoder_target_ids.shape[0]):
                if convert_decoder_target_ids[i].item() in ignore_index_li_with_number:
                    convert_decoder_target_ids[i] = -100

            # similar to https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/models/t5/modeling_t5.py#L1693
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(
                convert_logits,
                convert_decoder_target_ids
            )
            ret["loss"] = loss
        return ret




    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        strategy = self.config["training"].get("strategy", "ddp")
        if isinstance(strategy, dict) and "deepspeed" in strategy["class"].lower():
            if strategy["kwargs"].get("offload_optimizer", False):
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                print("Using DeepSpeedCPUAdam Optimizer for DeepSpeed Strategy due to offload_optimizer=True")
                optimizer = DeepSpeedCPUAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
            else:
                from deepspeed.ops.adam import FusedAdam
                print("Using FusedAdam Optimizer for DeepSpeed Strategy")
                optimizer = FusedAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["training"]["optimizer"]["lr"],
                weight_decay=self.config["training"]["optimizer"]["weight_decay"],
            )
        return optimizer


class OFAModuleCurriculum4(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Build tokenizer
        self.tokenizer = OFATokenizer.from_pretrained(
            config["model"]["ckpt_dir"]
        )
        _prevlen = len(self.tokenizer)
        self.tokenizer = add_space_to_tokenizer(self.tokenizer)
        print(f"Added space to tokenizer for codegen, new length: {_prevlen} -> {len(self.tokenizer)}")
        # NOTE: since we added these special spaces,
        # we need to use spaces_between_special_tokens=True when decoding, otherwise, we will get extra spaces between these
        # special "spaces" and the tokens around them.
        # https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/tokenization_utils.py#L951

        # 2. Build model
        self.model = OFAModel.from_pretrained(config["model"]["ckpt_dir"])
        # resize token embeddings
        # this will extend both self.model.encoder.embed_tokens and self.model.decoder.embed_tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # this will rebuild self.model.decoder.output_projection.weight
        # and use embed_tokens.weight as the initial value
        assert self.model.config.share_decoder_input_output_embed
        self.model.decoder.build_output_projection(self.model.config)

        self.save_hyperparameters()

    def forward_one_step(self, batch, object=True):
        ignore_li = ignore_index_li_with_number if object else ignore_index_li

        encoder_input_ids = batch["encoder_input_ids"]
        decoder_input_ids = batch["decoder_input_ids"]
        patch_images = batch["patch_images"]
        patch_masks = batch["patch_masks"]

        # non-pad positions are 1, pad positions are 0
        decoder_attention_mask = decoder_input_ids.ne(
            self.tokenizer.pad_token_id
        )
        # Forward
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=encoder_input_ids,
            patch_images=patch_images,
            patch_images_2=None,
            patch_masks=patch_masks,
            decoder_input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask
        )
        logits = outputs.logits
        # Calculate loss if decoder_target_ids is provided
        if "decoder_target_ids" in batch:
            convert_logits = logits.view(-1, logits.size(-1))

            # replace pad_token_id with -100 to ignore loss in decoder_target_ids
            decoder_target_ids = batch["decoder_target_ids"]
            convert_decoder_target_ids = decoder_target_ids.view(-1)
            for i in range(convert_decoder_target_ids.shape[0]):
                if convert_decoder_target_ids[i].item() in ignore_li:
                    convert_decoder_target_ids[i] = -100

            # similar to https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/models/t5/modeling_t5.py#L1693
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(
                convert_logits,
                convert_decoder_target_ids
            )
            # ret["loss"] = loss
        return (outputs, loss)









    def forward(self, batch):
        object, relation = batch['object'], batch['relation']
        _, loss_object = self.forward_one_step(object, True)
        _, loss_relation = self.forward_one_step(relation, False)
        total_loss = loss_object + loss_relation
        ret = {'loss': total_loss}
        return ret





    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        strategy = self.config["training"].get("strategy", "ddp")
        if isinstance(strategy, dict) and "deepspeed" in strategy["class"].lower():
            if strategy["kwargs"].get("offload_optimizer", False):
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                print("Using DeepSpeedCPUAdam Optimizer for DeepSpeed Strategy due to offload_optimizer=True")
                optimizer = DeepSpeedCPUAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
            else:
                from deepspeed.ops.adam import FusedAdam
                print("Using FusedAdam Optimizer for DeepSpeed Strategy")
                optimizer = FusedAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["training"]["optimizer"]["lr"],
                weight_decay=self.config["training"]["optimizer"]["weight_decay"],
            )
        return optimizer



class OFAModuleCurriculum5(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Build tokenizer
        self.tokenizer = OFATokenizer.from_pretrained(
            config["model"]["ckpt_dir"]
        )
        _prevlen = len(self.tokenizer)
        self.tokenizer = add_space_to_tokenizer(self.tokenizer)
        print(f"Added space to tokenizer for codegen, new length: {_prevlen} -> {len(self.tokenizer)}")
        # NOTE: since we added these special spaces,
        # we need to use spaces_between_special_tokens=True when decoding, otherwise, we will get extra spaces between these
        # special "spaces" and the tokens around them.
        # https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/tokenization_utils.py#L951

        # 2. Build model
        self.model = OFAModel.from_pretrained(config["model"]["ckpt_dir"])
        # resize token embeddings
        # this will extend both self.model.encoder.embed_tokens and self.model.decoder.embed_tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # this will rebuild self.model.decoder.output_projection.weight
        # and use embed_tokens.weight as the initial value
        assert self.model.config.share_decoder_input_output_embed
        self.model.decoder.build_output_projection(self.model.config)

        self.save_hyperparameters()

    def forward_one_step(self, batch, object=True):
        ignore_li = ignore_index_li_with_number if object else ignore_index_li

        encoder_input_ids = batch["encoder_input_ids"]
        decoder_input_ids = batch["decoder_input_ids"]
        patch_images = batch["patch_images"]
        patch_masks = batch["patch_masks"]

        # non-pad positions are 1, pad positions are 0
        decoder_attention_mask = decoder_input_ids.ne(
            self.tokenizer.pad_token_id
        )
        # Forward
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=encoder_input_ids,
            patch_images=patch_images,
            patch_images_2=None,
            patch_masks=patch_masks,
            decoder_input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask
        )
        logits = outputs.logits
        # Calculate loss if decoder_target_ids is provided
        if "decoder_target_ids" in batch:
            convert_logits = logits.view(-1, logits.size(-1))

            # replace pad_token_id with -100 to ignore loss in decoder_target_ids
            decoder_target_ids = batch["decoder_target_ids"]
            convert_decoder_target_ids = decoder_target_ids.view(-1)
            for i in range(convert_decoder_target_ids.shape[0]):
                if convert_decoder_target_ids[i].item() in ignore_li:
                    convert_decoder_target_ids[i] = -100

            # similar to https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/models/t5/modeling_t5.py#L1693
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(
                convert_logits,
                convert_decoder_target_ids
            )
            # ret["loss"] = loss
        return (outputs, loss)









    def forward(self, batch):
        object, relation, activity = batch['object'], batch['relation'], batch['activity']
        _, loss_object = self.forward_one_step(object, True)
        _, loss_relation = self.forward_one_step(relation, False)
        _, loss_activity = self.forward_one_step(activity, False)
        total_loss = loss_object + loss_relation + loss_activity
        ret = {'loss': total_loss}
        return ret





    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        strategy = self.config["training"].get("strategy", "ddp")
        if isinstance(strategy, dict) and "deepspeed" in strategy["class"].lower():
            if strategy["kwargs"].get("offload_optimizer", False):
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                print("Using DeepSpeedCPUAdam Optimizer for DeepSpeed Strategy due to offload_optimizer=True")
                optimizer = DeepSpeedCPUAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
            else:
                from deepspeed.ops.adam import FusedAdam
                print("Using FusedAdam Optimizer for DeepSpeed Strategy")
                optimizer = FusedAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["training"]["optimizer"]["lr"],
                weight_decay=self.config["training"]["optimizer"]["weight_decay"],
            )
        return optimizer




class OFAModuleMix(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Build tokenizer
        self.tokenizer = OFATokenizer.from_pretrained(
            config["model"]["ckpt_dir"]
        )
        _prevlen = len(self.tokenizer)
        self.tokenizer = add_space_to_tokenizer(self.tokenizer)
        print(f"Added space to tokenizer for codegen, new length: {_prevlen} -> {len(self.tokenizer)}")
        # NOTE: since we added these special spaces,
        # we need to use spaces_between_special_tokens=True when decoding, otherwise, we will get extra spaces between these
        # special "spaces" and the tokens around them.
        # https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/tokenization_utils.py#L951

        # 2. Build model
        self.model = OFAModel.from_pretrained(config["model"]["ckpt_dir"])
        # resize token embeddings
        # this will extend both self.model.encoder.embed_tokens and self.model.decoder.embed_tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # this will rebuild self.model.decoder.output_projection.weight
        # and use embed_tokens.weight as the initial value
        assert self.model.config.share_decoder_input_output_embed
        self.model.decoder.build_output_projection(self.model.config)

        self.save_hyperparameters()

    def forward(self, batch):
        surface, semantic = batch['surface'], batch['semantic']
        _, loss_surface = self.forward_once(surface)
        _, loss_semantic = self.forward_once(semantic)
        # detection, attribute, relation, activity = batch['detection'], batch['attribute'], batch['scenegraph'], batch['activity']
        # _, loss_detection = self.forward_once(detection)
        # _, loss_attribute = self.forward_once(attribute)
        # _, loss_relation = self.forward_once(relation)
        # _, loss_activity = self.forward_once(activity)
        total_loss = loss_surface + loss_semantic
        # total_loss = loss_detection + loss_attribute + loss_relation + loss_activity
        ret = {'loss': total_loss}
        return ret




    def forward_once(self, batch):
        encoder_input_ids = batch["encoder_input_ids"]
        decoder_input_ids = batch["decoder_input_ids"]
        patch_images = batch["patch_images"]
        patch_masks = batch["patch_masks"]

        # non-pad positions are 1, pad positions are 0
        decoder_attention_mask = decoder_input_ids.ne(
            self.tokenizer.pad_token_id
        )
        # Forward
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=encoder_input_ids,
            patch_images=patch_images,
            patch_images_2=None,
            patch_masks=patch_masks,
            decoder_input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask
        )
        logits = outputs.logits
        # Calculate loss if decoder_target_ids is provided
        if "decoder_target_ids" in batch:
            # replace pad_token_id with -100 to ignore loss in decoder_target_ids
            decoder_target_ids = batch["decoder_target_ids"]
            decoder_target_ids[
                decoder_target_ids == self.tokenizer.pad_token_id
            ] = -100
            # similar to https://github.com/huggingface/transformers/blob/3be028bc9d4b2cce9539b940f17052f333284684/src/transformers/models/t5/modeling_t5.py#L1693
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(
                logits.view(-1, logits.size(-1)),
                decoder_target_ids.view(-1)
            )
            # ret["loss"] = loss
        return (outputs, loss)




    def training_step(self, batch, batch_idx):
        # print(batch_idx)
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train_loss", loss)
        return loss





    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        strategy = self.config["training"].get("strategy", "ddp")
        if isinstance(strategy, dict) and "deepspeed" in strategy["class"].lower():
            if strategy["kwargs"].get("offload_optimizer", False):
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                print("Using DeepSpeedCPUAdam Optimizer for DeepSpeed Strategy due to offload_optimizer=True")
                optimizer = DeepSpeedCPUAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
            else:
                from deepspeed.ops.adam import FusedAdam
                print("Using FusedAdam Optimizer for DeepSpeed Strategy")
                optimizer = FusedAdam(
                    self.parameters(),
                    lr=self.config["training"]["optimizer"]["lr"],
                    weight_decay=self.config["training"]["optimizer"]["weight_decay"],
                )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["training"]["optimizer"]["lr"],
                weight_decay=self.config["training"]["optimizer"]["weight_decay"],
            )
        return optimizer