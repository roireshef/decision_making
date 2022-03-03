import os
import pickle
import re

import ray
import torch
from ray.rllib.agents import Trainer
from ray.rllib.agents.trainer import logger
from ray.rllib.models.modelv2 import ModelV2


class TorchPolicyUtils:
    """ Utility class that serializes and deserializes PyTorch-based Trainer object (and saves/loads to disk) """
    @staticmethod
    def load_model_params(trainer: Trainer, params_path: str, key_regex: str = ""):
        """ loads pytorch state_dict from file, filters by keys (nn.Modules names) according to key_regex and publishes
        to all workers of the trainer - local and remote """
        model: ModelV2 = trainer.workers.local_worker().get_policy().model
        loaded_state_dict: dict = torch.load(params_path, map_location=list(model.parameters())[0].device)

        # filter dictionary keys by regex
        rw = re.compile(key_regex)
        loaded_state_dict_filtered = dict(filter(lambda item: rw.match(item[0]), loaded_state_dict.items()))

        # complete skipped params from the local worker
        updated_state_dict = model.state_dict()
        updated_state_dict.update(loaded_state_dict_filtered)

        # load into local worker
        model.load_state_dict(updated_state_dict)

        # load into remote workers
        num_workers = len(trainer.workers.remote_workers())
        remote_state_dict = ray.put(updated_state_dict)
        logger.info("TorchPolicyUtils.load_model_params(trainer, %s, %s) is executing on %s workers" %
                    (params_path, key_regex, num_workers))
        call_ids = [rw.for_policy.remote(lambda p: p.set_weights(ray.get(remote_state_dict)))
                    for rw in trainer.workers.remote_workers()]
        ready_ids, _ = ray.wait(call_ids, num_returns=len(call_ids), timeout=10.)
        logger.info("TorchPolicyUtils.load_model_params(trainer, %s, %s) completed on %s/%s workers" %
                    (params_path, key_regex, len(ready_ids), num_workers))

    @staticmethod
    def set_requires_grad(trainer: Trainer, key_regex: str, requires_grad: bool):
        """ changes requires_grad attribute of all params of policy models that match key_regex in all workers of the
        trainer - local and remote """
        def modify(model):
            r = re.compile(key_regex)
            for key, param in model.named_parameters():
                if r.match(key):
                    param.requires_grad = requires_grad

        # change requires_grad locally
        modify(trainer.workers.local_worker().get_policy()._model)

        # change requires_grad remotely and wait for all workers to complete
        num_workers = len(trainer.workers.remote_workers())
        logger.info("TorchPolicyUtils.set_requires_grad(trainer, %s, %s) is executing on %s workers" %
                    (key_regex, requires_grad, num_workers))
        call_ids = [rw.for_policy.remote(lambda p: modify(p._model)) for rw in trainer.workers.remote_workers()]
        ready_ids, _ = ray.wait(call_ids, num_returns=len(call_ids), timeout=10.)
        logger.info("TorchPolicyUtils.set_requires_grad(trainer, %s, %s) completed on %s/%s workers" %
                    (key_regex, requires_grad, len(ready_ids), num_workers))

    @staticmethod
    def freeze_params(trainer: Trainer, key_regex: str):
        TorchPolicyUtils.set_requires_grad(trainer, key_regex, False)

    @staticmethod
    def unfreeze_params(trainer: Trainer, key_regex: str):
        TorchPolicyUtils.set_requires_grad(trainer, key_regex, True)


class SerializableTorchTrainerMixin(object):
    """ A Trainer Mixin for checkpointing pytorch policies """
    def save_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Save Policy Model parameters
        model: ModelV2 = self.workers.local_worker().get_policy().model
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "checkpoint.torch"))

        # Save Trainer counters
        trainer_info = self.train_exec_impl.shared_metrics.get().save()
        with open(os.path.join(checkpoint_path, "checkpoint.trainer_info"), "wb") as f:
            pickle.dump(trainer_info, f)

        trainable_state = self.get_state()
        with open(os.path.join(checkpoint_path, ".tune_metadata"), "wb") as f:
            trainable_state["saved_as_dict"] = False
            pickle.dump(trainable_state, f)

        open(os.path.join(checkpoint_path, ".is_checkpoint"), "a").close()

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        if checkpoint_path.endswith("checkpoint.torch"):
            checkpoint_path = checkpoint_path.replace("checkpoint.torch", "")

        # Load Policy Model
        TorchPolicyUtils.load_model_params(self, os.path.join(checkpoint_path, "checkpoint.torch"))

        # Load Trainer Counters
        try:
            with open(os.path.join(checkpoint_path, "checkpoint.trainer_info"), "rb") as f:
                train_exec_impl = pickle.load(f)

        except FileNotFoundError as e:
            # Keep for backward compatibility
            with open(os.path.join(checkpoint_path, "checkpoint.optimizer"), "rb") as f:
                train_exec_impl = {
                    "counters": pickle.load(f),
                    "info": {},
                    "timers": None
                }

        self.train_exec_impl.shared_metrics.get().restore(train_exec_impl)
