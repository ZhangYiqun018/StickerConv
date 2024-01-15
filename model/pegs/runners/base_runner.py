import os
import sys
import time
import datetime
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_scheduler

from utils import MergedConfig, MergedDatasetConfig
from pegs.model.pegs_modeling import PEGS
from pegs.optim import SchedulerType
from pegs.register import registry


@registry.register_runner("base_runner")
class BaseRunner:
    def __init__(self, config: MergedConfig) -> None:
        self.config = config
        
        self.model = self.init_model()
        self.set_model_to_device()

        self.datasets = self.build_datasets()
        self.dataloaders = self.create_dataloaders()
        split_name = self.dataloaders.keys()
        self.train_dataloader = self.dataloaders["train"]
        self.eval_dataloader = self.dataloaders["eval"] if "eval" in split_name else None
        
        self.iters_per_epoch = self.get_iters_per_epoch()
        
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()
        
        self.current_time = datetime.datetime.now().strftime("%b%d-%H-%M-%S")
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.run_config.outputs_dir, "runs", self.current_time))
        self.global_step = 0
        
    @property
    def model_config(self):
        return self.config.model
    
    @property
    def datasets_config(self):
        return self.config.datasets
    
    @property
    def run_config(self):
        return self.config.run
    
    @property
    def device(self):
        return self.run_config.device
    
    @property
    def use_distributed(self):
        return self.run_config.distributed
    
    @property
    def scaler(self):
        amp = self.run_config.get("amp", False)

        if amp:
            return torch.cuda.amp.GradScaler()
        else:
            return None
    
    @property
    def checkpoint_path(self):
        return self.model_config.get("checkpoint", None)
    
    @property
    def outputs_dir(self):
        return self.run_config.outputs_dir
    
    def init_model(self):
        return registry.get_model_class(self.model_config.arch)(**self.model_config)
    
    def set_model_to_device(self):
        # move model to device
        self.model = self.model.to(self.device)
        if hasattr(self.model, 'stable_diffusion'):
            self.model.stable_diffusion.to(self.device)
        # distributed training wrapper
        if self.use_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.run_config.gpu],
                find_unused_parameters=self.run_config.get("find_unused_parameters", False)
            )
    
    def unwrap_distributed_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model
    
    def build_datasets(self):
        datasets = dict()
        
        assert len(self.datasets_config) > 0, "At least one dataset has to be specified."
        
        for name in self.datasets_config:
            dataset_config = self.datasets_config[name]
            dataset_config = MergedDatasetConfig(dataset_config)

            if dataset_config.get("use", False) is False:
                continue
            
            builder = registry.get_builder_class(name)(dataset_config)
            logging.info(f"builder = {builder}")
            dataset = builder.build_dataset()  # {'train': }

            dataset["train"].name = name
            logging.info(f"dataset = {dataset}")
            if dataset_config.get("sample_ratio", None) is not None:
                dataset["train"].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset
        
        return datasets
    
    def create_dataloaders(self, eval: bool = False):
        # create dataloaders
        split_datasets = self.datasets[list(self.datasets.keys())[0]]
        split_names = sorted(split_datasets.keys())
        logging.info(f"split_names = {split_names}")

        datasets = [split_datasets[split] for split in split_names]

        batch_sizes = [
            self.run_config.batch_size_train
            if split == "train"
            else self.run_config.batch_size_eval
            for split in split_names
        ]
        
        loaders = []
        for dataset, batch_size in zip(datasets, batch_sizes):
            from webdataset import DataPipeline
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False if isinstance(dataset, DataPipeline) else True,
                num_workers=self.run_config.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
            )
            loaders.append(loader)
        dataloaders = {k: v for k, v in zip(split_names, loaders)} 
        
        return dataloaders

    def get_iters_per_epoch(self):
        if self.run_config.get("iters_per_epoch", None):
            return self.run_config.iters_per_epoch
        else:    
            try:
                iters_per_epoch = len(self.train_dataloader)
            except (AttributeError, TypeError):
                iters_per_epoch = 10000
            return iters_per_epoch
        
    @property
    def num_train_epochs(self):
        return self.run_config.num_train_epochs
    
    def init_optimizer(self):
        pg = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            params=pg,
            lr=self.run_config.learning_rate,
            betas=(self.run_config.beta1, self.run_config.beta2),
            eps=self.run_config.eps,
            weight_decay=self.run_config.weight_decay
        )
        
        return optimizer
    
    def init_lr_scheduler(self):
        if self.run_config.lr_scheduler in SchedulerType.keys():
            scheduler_cls = SchedulerType[self.run_config.lr_scheduler]
            if isinstance(scheduler_cls, str):
                lr_scheduler = get_scheduler(
                    scheduler_cls,
                    optimizer=self.optimizer,
                    num_warmup_steps=self.run_config.warmup_steps,
                    num_training_steps=self.iters_per_epoch * self.num_train_epochs
                )
            else:
                lr_scheduler = scheduler_cls(
                    optimizer=self.optimizer,
                    num_warmup_steps=self.run_config.warmup_steps,
                    num_training_steps=self.iters_per_epoch * self.num_train_epochs
                )
                
        else:
            raise ValueError("The lr_scheduler in config is not supported by transformers.")
        
        return lr_scheduler
        
    def train_one_epoch(
        self, 
        current_epoch: int
    ):
        dataloader = self.train_dataloader
        sum_loss = 0.0
        dataloader = tqdm(dataloader, file=sys.stdout)
        logging.info(f"iters_per_epoch = {self.iters_per_epoch}")
        for step, data in enumerate(dataloader):
            if step >= self.iters_per_epoch:
                break

            outputs = self.model(**data)
            loss = outputs.loss
            loss.backward()

            if self.model_config.enable_generation:
                self.model.module.reset_embeddings()
            
            # tensorboard
            self.tb_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.global_step)
            for key, value in outputs.items():
                self.tb_writer.add_scalar(key, value, self.global_step)
            self.global_step += 1
            
            sum_loss += loss.detach()
            avg_loss = sum_loss.item() / (step + 1)
        
            dataloader.desc = "[train epoch {}] loss: {:.3f}".format(current_epoch, avg_loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.lr_scheduler.step()

        logging.info("[train epoch {}] average loss: {:.3f}".format(current_epoch, avg_loss))
    
    def train(self):
        start_time = time.time()
        
        if self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)
        
        self.model.train()
        logging.info("Start training")
        self._get_parameter_number()
        
        for epoch in range(self.num_train_epochs):
            self.train_one_epoch(epoch)
            self.save_chechpoints(epoch)
        
            if self.use_distributed:
                distributed.barrier()
        
        cost_time = time.time() - start_time
        cost_time_str = str(datetime.timedelta(seconds=int(cost_time)))
        logging.info("Training time {}".format(cost_time_str))
        
    def save_chechpoints(self, current_epoch: int):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_distributed_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }

        state_dict = model_no_ddp.state_dict()

        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
                
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.config),  # OmegaConf to Dict
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": current_epoch,
        }
        
        save_checkpoint_path = os.path.join(
            self.outputs_dir,
            "checkpoint_{}_{}.pth".format(self.current_time, current_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(current_epoch, save_checkpoint_path))
        torch.save(save_obj, save_checkpoint_path)
        
    def evaluate(self, current_epoch):
        pass
        
    def _load_checkpoint(self, filename):
        """
        Resume from a checkpoint/checkpoints.
        """
        if isinstance(filename, str):
            if os.path.isfile(filename):
                checkpoint = torch.load(filename, map_location=self.device)
                state_dict = checkpoint["model"]
                self.unwrap_distributed_model(self.model).load_state_dict(state_dict, strict=False)
            else:
                raise ValueError("checkpoint path is invalid!")
        elif isinstance(filename, ListConfig):
            for one_filename in filename:
                checkpoint = torch.load(one_filename, map_location=self.device)
                state_dict = checkpoint["model"]
                self.unwrap_distributed_model(self.model).load_state_dict(state_dict, strict=False)
        
        logging.info("Resume checkpoint from {}".format(filename))
        
    def _get_parameter_number(self):
        """
        Number of model parameters (without stable diffusion)
        """
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_ratio = trainable_num / total_num
        
        logging.info(
            "\nparams:\n[total]: {}\n[trainable]: {}\n[ratio]:{}".format(
                total_num, trainable_num, trainable_ratio
            )
        )
    
