import json
import os
import random
import warnings

import numpy as np
import paddle
import paddle.distributed as dist
from hnas.utils.alphanet_loss import AdaptiveLossSoft
from hnas.utils.compute_zen_score import compute_nas_score
from hnas.utils.flops_calculation import get_arch_flops
from hnas.utils.pairwise_rank import PairwiseRankLoss
from model_zennas import Model as myModel
from paddle import Model, fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.distributed import ParallelEnv
from paddle.fluid.executor import global_scope
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.distributed import collective
from paddle import flatten
from paddle.hapi.callbacks import EarlyStopping, config_callbacks
from paddle.hapi.model import DynamicGraphAdapter
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddleslim.nas.ofa import OFA

from utils.distance_calc import hamming_distance

from ..dataset.dataiter import DataLoader as TrainDataLoader
from ..dataset.random_size_crop import MyRandomResizedCrop


def _all_gather(x, nranks, ring_id=0, use_calc_stream=True):
    return collective._c_allgather(
        x, nranks, ring_id=ring_id, use_calc_stream=use_calc_stream)


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def to_numpy(var):
    assert isinstance(var, (Variable, fluid.core.VarBase)), "not a variable"
    if isinstance(var, fluid.core.VarBase):
        return var.numpy()
    t = global_scope().find_var(var.name).get_tensor()
    return np.array(t)


def _update_input_info(inputs):
    "Get input shape list by given inputs in Model initialization."
    shapes = None
    dtypes = None
    if isinstance(inputs, list):
        shapes = [list(input.shape) for input in inputs]
        dtypes = [input.dtype for input in inputs]
    elif isinstance(inputs, dict):
        shapes = [list(inputs[name].shape) for name in inputs]
        dtypes = [inputs[name].dtype for name in inputs]
    else:
        return None
    return shapes, dtypes


class MyDynamicGraphAdapter(DynamicGraphAdapter):
    def __init__(self, model, cfg=None):
        self.model = model
        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank
        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0
        }

        self._input_info = None
        if self._nranks > 1:
            dist.init_parallel_env()
            if isinstance(self.model.network, OFA):
                self.model.network.model = paddle.DataParallel(self.model.network.model, find_unused_parameters=True)
                self.ddp_model = self.model.network
            else:
                self.ddp_model = paddle.DataParallel(self.model.network)
        self.dyna_bs = cfg.get('dynamic_batch_size', 1)

        self.pairwise_rankloss = PairwiseRankLoss()

    # TODO multi device in dygraph mode not implemented at present time
    def train_batch(self, inputs, labels=None, **kwargs):
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode = 'train'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        labels = [to_variable(l) for l in to_list(labels)]
        epoch = kwargs.get('epoch', None)
        self.epoch = epoch
        nBatch = kwargs.get('nBatch', None)
        step = kwargs.get('step', None)
        for i in range(self.dyna_bs):
            subnet_seed = int('%d%.1d' % (epoch * nBatch + step, i))
            np.random.seed(subnet_seed)

            # once for all
            current_config = self.model.network._progressive_shrinking("random")
            self.model.network.set_net_config(current_config)

            # print(self.model.network.gen_subnet_code)
            if self._nranks > 1:
                outputs = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                outputs = self.model.network.forward(*[to_variable(x) for x in inputs])

            # change this place to process the output of network 
            losses = self.model._loss(*(to_list(outputs) + labels))
            losses = to_list(losses)
            final_loss = fluid.layers.sum(losses)
            final_loss.backward()

        self.model._optimizer.step()
        self.model._optimizer.clear_grad()

        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.compute(*(to_list(outputs) + labels))
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return ([to_numpy(l) for l in losses], metrics) if len(metrics) > 0 else [to_numpy(l) for l in losses]

    # TODO multi device in dygraph mode not implemented at present time
    def train_batch_sandwich_with_rank(self, inputs, labels=None, **kwargs):
        # follow sandwich rule in autoslim flops guided 
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode = 'train'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = to_variable(labels).squeeze(0)
        epoch = kwargs.get('epoch', None)
        self.epoch = epoch
        nBatch = kwargs.get('nBatch', None)
        step = kwargs.get('step', None)

        # set seed 
        subnet_seed = int('%d%.1d' % (epoch * nBatch + step, step))
        np.random.seed(subnet_seed)

        # sample largest subnet as teacher net 
        largest_config = self.model.network.active_autoslim_subnet(sample_type="largest")
        self.model.network.set_net_config(largest_config)
        if self._nranks > 1:
            teacher_output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
        else:
            teacher_output = self.model.network.forward(*[to_variable(x) for x in inputs])
        ### normal forward with gt 
        loss1 = self.model._loss(input=teacher_output[0], tea_input=None, label=labels)
        loss1.backward()

        # sample smallest subnet as student net and perform distill operation
        smallest_config = self.model.network.active_autoslim_subnet(sample_type="smallest")
        self.model.network.set_net_config(smallest_config)
        ### forward with inplace distillation
        if self._nranks > 1:
            output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
        else:
            output = self.model.network.forward(*[to_variable(x) for x in inputs])

        loss2 = self.model._loss(input=output[0],tea_input=teacher_output[0], label=None)
        loss2.backward()
        del output 

        # sample random subnets as student net and perform distill operation
        for _ in range(self.dyna_bs-2): # 4
            random_config1 = self.model.network.active_autoslim_subnet(sample_type="random")
            self.model.network.set_net_config(random_config1)
            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            
            loss = self.model._loss(input=output[0],tea_input=teacher_output[0], label=None)
            loss.backward()
            del output

        for _ in range(2):
            # generate two sample to compare with each other 
            random_config1 = self.model.network.active_autoslim_subnet(sample_type="random")
            self.model.network.set_net_config(random_config1)
            flops1 = get_arch_flops(self.model.network.gen_subnet_code)
            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            loss3 = self.model._loss(input=output[0],tea_input=None, label=labels)
            del output

            random_config2 = self.model.network.active_autoslim_subnet(sample_type="random")
            self.model.network.set_net_config(random_config2)
            flops2 = get_arch_flops(self.model.network.gen_subnet_code)
            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            loss4 = self.model._loss(input=output[0],tea_input=None, label=labels)
            
            # loss5 =  2 * np.sin(np.pi * 0.8 * epoch / 70) * self.pairwise_rankloss(flops1, flops2, loss3, loss4)
            loss5 =  min(2, epoch/10.) * self.pairwise_rankloss(flops1, flops2, loss3, loss4)
            loss5.backward()
            
        self.model._optimizer.step()
        self.model._optimizer.clear_grad()

        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.compute(output[0], labels)
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return ([to_numpy(l) for l in [loss1]], metrics) if len(metrics) > 0 else [to_numpy(l) for l in [loss1]]

    def train_batch_sandwich_with_zenscore(self, inputs, labels=None, **kwargs):
         # follow sandwich rule in autoslim
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode = 'train'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = to_variable(labels).squeeze(0)
        epoch = kwargs.get('epoch', None)
        self.epoch = epoch
        nBatch = kwargs.get('nBatch', None)
        step = kwargs.get('step', None)

        # set seed 
        subnet_seed = int('%d%.1d' % (epoch * nBatch + step, step))
        np.random.seed(subnet_seed)

        # sample largest subnet as teacher net 
        largest_config = self.model.network.active_autoslim_subnet(sample_type="largest")
        self.model.network.set_net_config(largest_config)
        if self._nranks > 1:
            teacher_output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
        else:
            teacher_output = self.model.network.forward(*[to_variable(x) for x in inputs])
        ### normal forward with gt 
        loss1 = self.model._loss(input=teacher_output[0], tea_input=None, label=labels)
        loss1.backward()

        # sample smallest subnet as student net and perform distill operation
        smallest_config = self.model.network.active_autoslim_subnet(sample_type="smallest")
        self.model.network.set_net_config(smallest_config)
        ### forward with inplace distillation
        if self._nranks > 1:
            output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
        else:
            output = self.model.network.forward(*[to_variable(x) for x in inputs])

        loss2 = self.model._loss(input=output[0],tea_input=teacher_output[0], label=None)
        loss2.backward()
        del output 


        # sample random subnets as student net and perform distill operation
        for _ in range(self.dyna_bs-2): 
            random_config1 = self.model.network.active_autoslim_subnet(sample_type="random")
            self.model.network.set_net_config(random_config1)
            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            
            loss = self.model._loss(input=output[0],tea_input=teacher_output[0], label=None)
            loss.backward()
            del output

        for _ in range(self.dyna_bs - 1):
            # generate two sample to compare with each other 
            random_config1 = self.model.network.active_autoslim_subnet(sample_type="random")
            self.model.network.set_net_config(random_config1)
            anchor_arch = self.model.network.gen_subnet_code

            # flops1 = get_arch_flops(self.model.network.gen_subnet_code)
            model1 = myModel(arch=self.model.network.gen_subnet_code, block='basic')
            avg_nas_score1, std_nas_score1, avg_precision1 = compute_nas_score(
                model1, 
                batch_size=16,
                resolution=224, 
                repeat=32
            )

            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            loss3 = self.model._loss(input=output[0],tea_input=None, label=labels)
            del output

            dis = 0 
            while dis < 33: 
                random_config2 = self.model.network.active_autoslim_subnet(sample_type="random")
                self.model.network.set_net_config(random_config2)
                current_arch = self.model.network.gen_subnet_code 
                dis = hamming_distance(anchor_arch, current_arch)

            # flops2 = get_arch_flops(self.model.network.gen_subnet_code)
            model2 = myModel(arch=self.model.network.gen_subnet_code, block='basic')
            avg_nas_score2, std_nas_score2, avg_precision2 = compute_nas_score(
                model2, 
                batch_size=16,
                resolution=224, 
                repeat=32
            )
            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            loss4 = self.model._loss(input=output[0],tea_input=None, label=labels)

            loss5 = self.pairwise_rankloss(avg_nas_score1, avg_nas_score2, loss3, loss4)
            loss5.backward()

        self.model._optimizer.step()
        self.model._optimizer.clear_grad()

        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.compute(output[0], labels)
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return ([to_numpy(l) for l in [loss1]], metrics) if len(metrics) > 0 else [to_numpy(l) for l in [loss1]]


    # TODO multi device in dygraph mode not implemented at present time
    def train_batch_sandwich(self, inputs, labels=None, **kwargs):
        ALPHALOSS = False

        if ALPHALOSS:
            alpha = AdaptiveLossSoft()
        # follow sandwich rule in autoslim
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode = 'train'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = to_variable(labels).squeeze(0)
        epoch = kwargs.get('epoch', None)
        self.epoch = epoch
        nBatch = kwargs.get('nBatch', None)
        step = kwargs.get('step', None)

        # set seed 
        subnet_seed = int('%d%.1d' % (epoch * nBatch + step, step))
        np.random.seed(subnet_seed)

        # sample largest subnet as teacher net 
        largest_config = self.model.network.active_autoslim_subnet(sample_type="largest")
        self.model.network.set_net_config(largest_config)
        if self._nranks > 1:
            teacher_output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
        else:
            teacher_output = self.model.network.forward(*[to_variable(x) for x in inputs])
        ### normal forward with gt 
        loss1 = self.model._loss(input=teacher_output[0], tea_input=None, label=labels)
        loss1.backward()

        # sample smallest subnet as student net and perform distill operation
        smallest_config = self.model.network.active_autoslim_subnet(sample_type="smallest")
        self.model.network.set_net_config(smallest_config)
        ### forward with inplace distillation
        if self._nranks > 1:
            output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
        else:
            output = self.model.network.forward(*[to_variable(x) for x in inputs])

        if ALPHALOSS:
            loss2 = alpha(output[0], teacher_output[0])
        else:
            loss2 = self.model._loss(input=output[0],tea_input=teacher_output[0], label=None)
        loss2.backward()

        # sample random subnets as student net and perform distill operation
        for _ in range(self.dyna_bs-2): 
            random_config = self.model.network.active_autoslim_subnet(sample_type="random")
            self.model.network.set_net_config(random_config)
            if self._nranks > 1:
                output = self.ddp_model.forward(*[to_variable(x) for x in inputs])
            else:
                output = self.model.network.forward(*[to_variable(x) for x in inputs])
            if ALPHALOSS:
                loss3 = alpha(output[0], teacher_output[0])
            else:
                loss3 = self.model._loss(input=output[0],tea_input=teacher_output[0], label=None)
            loss3.backward()

        self.model._optimizer.step()
        self.model._optimizer.clear_grad()

        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.compute(output[0], labels)
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return ([to_numpy(l) for l in [loss1]], metrics) if len(metrics) > 0 else [to_numpy(l) for l in [loss1]]

    def eval_batch(self, inputs, labels=None):
        self.model.network.eval()
        self.model.network.model.eval()
        self.mode = 'eval'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        # labels = [to_variable(l) for l in to_list(labels)]

        outputs = self.model.network.forward(*[to_variable(x) for x in inputs])
        if self.model._loss:
            losses = self.model._loss(input=outputs, label=to_variable(labels).squeeze(0))
            losses = to_list(losses)

        if self._nranks > 1:
            outputs = [_all_gather(o, self._nranks) for o in to_list(outputs)]
            labels = [_all_gather(l, self._nranks) for l in labels]
        metrics = []
        for metric in self.model._metrics:
            # cut off padding value.
            if self.model._test_dataloader is not None and self._nranks > 1 \
                    and isinstance(self.model._test_dataloader, DataLoader):
                total_size = len(self.model._test_dataloader.dataset)
                samples = outputs[0].shape[0]
                current_count = self._merge_count.get(self.mode + '_total', 0)
                if current_count + samples >= total_size:
                    outputs = [
                        o[:int(total_size - current_count)] for o in outputs
                    ]
                    labels = [
                        l[:int(total_size - current_count)] for l in labels
                    ]
                    self._merge_count[self.mode + '_total'] = 0
                    self._merge_count[self.mode + '_batch'] = int(total_size -
                                                                  current_count)
                else:
                    self._merge_count[self.mode + '_total'] += samples
                    self._merge_count[self.mode + '_batch'] = samples

            metric_outs = metric.compute(*(to_list(outputs) + labels))
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        if self.model._loss and len(metrics):
            return [to_numpy(l) for l in losses], metrics
        elif self.model._loss:
            return [to_numpy(l) for l in losses]
        else:
            return metrics

    def save(self, path):
        params = self.model.network.state_dict()
        fluid.save_dygraph(params, path)
        if self.model._optimizer is None:
            return
        if self.model._optimizer.state_dict():
            optim = self.model._optimizer.state_dict()
            optim['epoch'] = self.epoch
            fluid.save_dygraph(optim, path)


class Trainer(Model):
    def __init__(self, network, inputs=None, labels=None, cfg=None):
        # super().__init__(network, inputs=inputs, labels=labels)

        self.mode = 'train'
        self.network = network
        self._inputs = None
        self._labels = None
        self._loss = None
        self._loss_weights = None
        self._optimizer = None
        self._input_info = None
        self._is_shape_inferred = False
        self._test_dataloader = None
        self.stop_training = False

        self._inputs = self._verify_spec(inputs, is_input=True)
        self._labels = self._verify_spec(labels)
        # init backend
        self._adapter = MyDynamicGraphAdapter(self, cfg)
        self.start_epoch = 0

    def fit(
            self,
            train_data=None,
            eval_data=None,
            batch_size=1,
            epochs=1,
            eval_freq=5,
            log_freq=10,
            save_dir=None,
            save_freq=1,
            verbose=2,
            drop_last=False,
            shuffle=True,
            num_workers=4,
            callbacks=None, ):
        assert train_data is not None, "train_data must be given!"

        if isinstance(train_data, Dataset):
            train_sampler = DistributedBatchSampler(
                train_data,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
            train_loader = TrainDataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        else:
            train_loader = train_data

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=batch_size)
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        elif eval_data is not None:
            eval_loader = eval_data
        else:
            eval_loader = None

        do_eval = eval_loader is not None
        self._test_dataloader = eval_loader

        steps = self._len_data_loader(train_loader)
        cbks = config_callbacks(
            callbacks,
            model=self,
            epochs=epochs,
            steps=steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(), )

        if any(isinstance(k, EarlyStopping) for k in cbks) and not do_eval:
            warnings.warn("EarlyStopping needs validation data.")

        cbks.on_begin('train')
        for epoch in range(self.start_epoch, epochs):
            cbks.on_epoch_begin(epoch)
            self.network.set_epoch(epoch)
            logs = self._run_one_epoch(train_loader, cbks, 'train', epoch=epoch)
            cbks.on_epoch_end(epoch, logs)

            if do_eval and epoch % eval_freq == 0:

                eval_steps = self._len_data_loader(eval_loader)
                cbks.on_begin('eval', {
                    'steps': eval_steps,
                    'metrics': self._metrics_name()
                })

                eval_logs = self._run_one_epoch(eval_loader, cbks, 'eval')

                cbks.on_end('eval', eval_logs)
                if self.stop_training:
                    break

        cbks.on_end('train', logs)
        self._test_dataloader = None

    def train_batch(self, inputs, labels=None, **kwargs):
        # call the function train_batch of adapter
        loss = self._adapter.train_batch(inputs, labels, **kwargs)
        if fluid.in_dygraph_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    def train_batch_sandwich(self, inputs, labels=None, **kwargs):
        # call the function train_batch of adapter
        loss = self._adapter.train_batch_sandwich_with_rank(inputs, labels, **kwargs)
        if fluid.in_dygraph_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    def _run_one_epoch(self, data_loader, callbacks, mode, logs={}, **kwargs):
        outputs = []
        if mode == 'train':
            MyRandomResizedCrop.epoch = kwargs.get('epoch', None)

        for step, data in enumerate(data_loader):
            print("data: ", data)
            data = flatten(paddle.to_tensor(data))

            batch_size = data[0].shape()[0] if callable(data[0].shape) else data[0].shape[0]

            callbacks.on_batch_begin(mode, step, logs)

            if mode != 'predict':
                if mode == 'train':
                    MyRandomResizedCrop.sample_image_size(step)
                    # call train_batch function
                    outs = getattr(self, mode + '_batch_sandwich')(data[:len(self._inputs)],
                                                          data[len(self._inputs):],
                                                          epoch=kwargs.get('epoch', None),
                                                          nBatch=len(data_loader),
                                                          step=step)

                else:
                    outs = getattr(self, mode + '_batch')(data[:len(self._inputs)], data[len(self._inputs):])
                if self._metrics and self._loss:
                    metrics = [[l[0] for l in outs[0]]]
                elif self._loss:
                    metrics = [[l[0] for l in outs]]
                else:
                    metrics = []

                # metrics
                for metric in self._metrics:
                    res = metric.accumulate()
                    metrics.extend(to_list(res))

                assert len(self._metrics_name()) == len(metrics)
                for k, v in zip(self._metrics_name(), metrics):
                    logs[k] = v
            else:
                if self._inputs is not None:
                    outs = self.predict_batch(data[:len(self._inputs)])
                else:
                    outs = self.predict_batch(data)

                outputs.append(outs)

            logs['step'] = step
            if mode == 'train' or self._adapter._merge_count.get(mode + '_batch', 0) <= 0:
                logs['batch_size'] = batch_size * ParallelEnv().nranks
            else:
                logs['batch_size'] = self._adapter._merge_count[mode + '_batch']

            callbacks.on_batch_end(mode, step, logs)
        self._reset_metrics()

        if mode == 'predict':
            return logs, outputs
        return logs

    def evaluate(
            self,
            eval_data,
            batch_size=1,
            log_freq=10,
            verbose=1,
            eval_sample_num=10,
            num_workers=0,
            callbacks=None):

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(eval_data, batch_size=batch_size)
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True, use_shared_memory=True)
        else:
            eval_loader = eval_data

        self._test_dataloader = eval_loader

        cbks = config_callbacks(
            callbacks,
            model=self,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(), )

        eval_steps = self._len_data_loader(eval_loader)

        self.network.model.eval()

        import time
        sample_result = []
        for i in range(eval_sample_num):
            cbks.on_begin('eval', {'steps': eval_steps, 'metrics': self._metrics_name()})
            subnet_seed = int(time.time() / 10)
            random.seed(subnet_seed)
            self.network.active_subnet(224)
            logs = self._run_one_epoch(eval_loader, cbks, 'eval')

            cbks.on_end('eval', logs)

            self._test_dataloader = None

            eval_result = {}
            for k in self._metrics_name():
                eval_result[k] = logs[k]
            sample_res = '{} {} {}'.format(
                self.network.gen_subnet_code, eval_result['acc_top1'], eval_result['acc_top5'])
            if ParallelEnv().local_rank == 0:
                print(sample_res)
            sample_result.append(sample_res)
            if ParallelEnv().local_rank == 0:
                with open('channel_sample.txt', 'a') as f:
                    f.write('{}\n'.format(sample_res))

        return sample_result

    def evaluate_whole_test(
            self,
            eval_data,
            batch_size=256,
            log_freq=10,
            verbose=1,
            num_workers=4,
            callbacks=None,
            json_path=None):

        candidate_path = json_path 

        with open(candidate_path, "r") as f:
            candidate_dict = json.load(f)
            save_candidate = candidate_dict.copy()

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = None 
            eval_loader = DataLoader(
                eval_data, 
                batch_sampler=eval_sampler,
                places=self._place,
                shuffle=False, 
                num_workers=num_workers,
                batch_size=batch_size, 
                return_list=True, 
                use_shared_memory=True,
                use_buffer_reader=True)
        else:
            eval_loader = eval_data

        self._test_dataloader = eval_loader

        cbks = config_callbacks(
            callbacks,
            model=self,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(), )

        eval_steps = self._len_data_loader(eval_loader)

        self.network.model.eval()

        import time
        show_flag = True

        sample_result = []
        for arch_name, config in candidate_dict.items():
            s1 = time.time() 
            cbks.on_begin('eval', {'steps': eval_steps, 'metrics': self._metrics_name()})

            self.network.active_specific_subnet(224, config['arch'])

            logs = self._run_one_epoch(eval_loader, cbks, 'eval')
            
            s3 = time.time()
            if ParallelEnv().local_rank == 0 and show_flag:
                print("forward_one_epoch time: ", s3-s1)

            cbks.on_end('eval', logs)

            self._test_dataloader = None

            eval_result = {}
            for k in self._metrics_name():
                eval_result[k] = logs[k]
            sample_res = '{} {} {} {}'.format(arch_name, config['arch'], eval_result['acc_top1'], eval_result['acc_top5'])
            if ParallelEnv().local_rank == 0:
                print(sample_res)

            sample_result.append(sample_res)
            save_candidate[arch_name]['acc'] = eval_result['acc_top1']

        if ParallelEnv().local_rank == 0:
            save_path = "./checkpoints/submit_results.json"
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
                
            with open(save_path, 'w') as f:
                print(f"save to {save_path}")
                json.dump(save_candidate, f)

        return sample_result
