import os
import random
import torch
import shutil
import numpy as np
from tqdm import tqdm
from lib.data.data_utils import transform_packed_sequence_multiple
from lib.distributed import (
    barrier,
    broadcast_object,
    distributed_is_initialized,
    is_main_process,
    load_model_state_dict,
    rank,
    reduce_mean,
    reduce_sum,
    unwrap_model,
)


TRAINING_STATE_FILENAME = 'training_state.pth'


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_rng_byte_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device='cpu')


def _restore_rng_state(state):
    if not state:
        return

    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(_as_cpu_rng_byte_tensor(state['torch']))
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            _as_cpu_rng_byte_tensor(cuda_state)
            for cuda_state in state['torch_cuda']
        ])


def save_training_state(checkpoint_path, model, optimizer, lr_scheduler, logger, epoch):
    checkpoint_path = os.fspath(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'epoch': int(epoch),
        'next_epoch': int(epoch) + 1,
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'best_epoch': getattr(logger, 'best_epoch', None) if logger else None,
        'train_loss': list(getattr(logger, 'train_loss', [])) if logger else [],
        'loss_evolution': list(getattr(logger, 'loss_evolution', [])) if logger else [],
        'rng_state': _capture_rng_state(),
    }
    tmp_path = f"{checkpoint_path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def load_training_state(checkpoint_path, model, optimizer, lr_scheduler, logger, map_location=None):
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    load_model_state_dict(model, state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    lr_scheduler.load_state_dict(state['scheduler_state_dict'])

    if logger:
        logger.train_loss = list(state.get('train_loss', []))
        logger.loss_evolution = list(state.get('loss_evolution', []))
        logger.best_epoch = state.get('best_epoch', -1)

    _restore_rng_state(state.get('rng_state'))
    return state


def _best_epoch_from_logger(logger, fallback=None):
    if not logger:
        return fallback
    best_epoch = getattr(logger, 'best_epoch', None)
    if best_epoch is not None and best_epoch >= 0:
        return best_epoch
    loss = getattr(logger, 'loss_evolution', [])
    if loss:
        return loss.index(min(loss))
    return fallback


def _reduce_validation_state(valid_loss_sum, valid_batch_count, logger, device):
    stats = [valid_loss_sum, valid_batch_count]
    if logger is not None:
        stats.extend([
            logger.mse,
            logger.mse1,
            logger.mse2,
            logger.mse3,
            logger.mse4,
            logger.iters_counted,
        ])
    stats = reduce_sum(stats, device)
    total_loss, total_batches = stats[:2]
    if logger is not None:
        (
            logger.mse,
            logger.mse1,
            logger.mse2,
            logger.mse3,
            logger.mse4,
            logger.iters_counted,
        ) = stats[2:]
        logger.iters_counted = int(logger.iters_counted)
    return total_loss / total_batches if total_batches else float('nan')


def train(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, cfg, start_epoch=0, checkpoint_path=None, best_epoch=None):
    start_epoch = int(start_epoch or 0)
    best_epoch = _best_epoch_from_logger(logger, fallback=best_epoch)
    try:
        for epoch in range(start_epoch, cfg.train.max_epochs):
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            if cfg.run_config.variable_sequence_length:
                train_loss = train_var_epoch(train_dataloader, encoder_forecaster, criterion,
                                        optimizer, wrf_scaler, era_scaler, cfg)

                train_loss = reduce_mean(train_loss, cfg.device)
                valid_loss = eval_var_epoch(
                    unwrap_model(encoder_forecaster), criterion, wrf_scaler,
                    era_scaler, valid_dataloader, logger, cfg,
                )
            else:
                train_loss = train_epoch(train_dataloader, encoder_forecaster, criterion,
                                        optimizer, wrf_scaler, era_scaler, cfg)

                train_loss = reduce_mean(train_loss, cfg.device)
                valid_loss = eval_epoch(
                    unwrap_model(encoder_forecaster), criterion, wrf_scaler,
                    era_scaler, valid_dataloader, logger, cfg,
                )
            if is_main_process():
                print('train loss', train_loss)
                print('valid_loss', valid_loss)
            barrier()
            lr_scheduler.step()
            should_stop = False
            if is_main_process():
                print(lr_scheduler.get_last_lr())
                logger.train_loss.append(train_loss)
                logger.print_stat_readable(epoch)
                best_epoch = logger.save_model(unwrap_model(encoder_forecaster).state_dict(), epoch)
                if checkpoint_path is not None:
                    save_training_state(checkpoint_path, encoder_forecaster, optimizer, lr_scheduler, logger, epoch)
                if epoch - best_epoch > 5:
                    should_stop = True
            elif logger:
                logger.reset_stat()
            if distributed_is_initialized():
                best_epoch, should_stop = broadcast_object((best_epoch, should_stop), src=0)
            if should_stop:
                break
    except KeyboardInterrupt:
        if checkpoint_path is not None:
            print(f"Training interrupted. Resume checkpoint is at {checkpoint_path}")
        raise
    # logger.save_configuration() if logger else None
    return best_epoch, encoder_forecaster


def train_epoch(dataloader, model, criterion, optimizer, wrf_scaler, era_scaler, cfg):
    train_loss = 0
    model.train()
    t = 0
    # for train_data, train_label, stations, scatter, dates in (pbar := tqdm(dataloader)):
    stations = scatter = None
    for data, dates in (pbar := tqdm(dataloader)):
        if data is None:
            print('Empty data batch, continue')
            continue
        train_data, train_label = data.pop(cfg.reference_dataset), data.pop(cfg.target_dataset)

        if train_data is None:
            print(dates, 'Train data is None, continue')
            continue

        train_data = torch.swapaxes(train_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
        train_data = wrf_scaler.transform(train_data, dims=2)
        train_label = torch.swapaxes(train_label.type(torch.float), 0, 1)
        train_label = train_label.flatten(-2, -1)[..., criterion.meaner.target_slice].to(cfg.device)
        train_label = era_scaler.transform(train_label, dims=2)

        if cfg.run_config.use_stations and 'Stations' in data:
            stations = data.pop('Stations')
            stations = torch.permute(stations.type(torch.float).to(cfg.device), (1, 0, 3, 2))
            stations = era_scaler.transform(stations, dims=2)
        
        scatter_data, scatter_times = None, None
        if cfg.run_config.use_scatter and 'Scatter' in data:
            scatter = data.pop('Scatter')
            scatter_times = scatter[0].to(cfg.device).type(torch.double)
            scatter_data = torch.stack((scatter[1], scatter[2]), dim=-3).type(torch.float).to(cfg.device)
            scatter_data = wrf_scaler.transform(scatter_data, dims=2,
                                                means=wrf_scaler.means[:2],
                                                stds=wrf_scaler.stddevs[:2])
            
        batch_dates = torch.as_tensor(dates.astype('datetime64[s]').astype('float64')).to(cfg.device)

        optimizer.zero_grad()
        output = model(train_data)
        train_data = train_data[:, :, :3]
        loss = criterion(train_data, output, train_label, stations,
                         scatter_data, scatter_times, batch_dates)

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_epoch(model, criterion, wrf_scaler, era_scaler, dataloader, logger, cfg):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        valid_batch_count = 0
        stations = scatter = None
        for data, dates in tqdm(dataloader, disable=rank() != 0):
            if data is None:
                if is_main_process():
                    print('Empty data batch, continue')
                continue
            valid_data, valid_label = data.pop(cfg.reference_dataset), data.pop(cfg.target_dataset)
            if valid_data is None:
                if is_main_process():
                    print(dates, 'Valid data is None, continue')
                continue        

            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            valid_data = wrf_scaler.transform(valid_data, dims=2)
            valid_label = torch.swapaxes(valid_label.type(torch.float), 0, 1)
            valid_label = valid_label.flatten(-2, -1)[..., criterion.meaner.target_slice].to(cfg.device)
            valid_label = era_scaler.transform(valid_label, dims=2)

            if cfg.run_config.use_stations and 'Stations' in data:
                stations = torch.permute(stations.type(torch.float).to(cfg.device), (1, 0, 3, 2))
                stations = era_scaler.transform(stations, dims=2)

            scatter_data, scatter_times = None, None
            if cfg.run_config.use_scatter and 'Scatter' in data:
                scatter_times = scatter[0].to(cfg.device).type(torch.double)
                scatter_data = torch.stack((scatter[1], scatter[2]), dim=-3).type(torch.float).to(cfg.device)
                scatter_data = wrf_scaler.transform(scatter_data, dims=2,
                                                    means=wrf_scaler.means[:2],
                                                    stds=wrf_scaler.stddevs[:2])
                
            batch_dates = torch.as_tensor(dates.astype('datetime64[s]').astype('float64')).to(cfg.device)

            output = model(valid_data)

            valid_data = valid_data[:, :, :3]
            loss = criterion(valid_data, output, valid_label, stations,
                             scatter_data, scatter_times, batch_dates, logger=logger)
            valid_loss += loss.item()
            valid_batch_count += 1

    return _reduce_validation_state(valid_loss, valid_batch_count, logger, cfg.device)


def train_var_epoch(dataloader, model, criterion, optimizer, wrf_scaler, era_scaler, cfg):
    train_loss = 0
    model.train()
    t = 0
    for train_data, train_label, stations, scatter, i in (pbar := tqdm(dataloader)):
        if train_data is None:
            continue
        train_data = transform_packed_sequence_multiple(train_data.to(cfg.device), [(torch.Tensor.type, (torch.float,), {}),
                                                                                    (wrf_scaler.transform, (), {'dims': 1})])
        train_label = transform_packed_sequence_multiple(train_label.to(cfg.device), [(torch.Tensor.type, (torch.float,), {}),
                                                                                      (era_scaler.transform, (), {'dims': 1})])

        # if stations is not None:
        #     stations = torch.permute(stations.type(torch.float).to(cfg.device), (1, 0, 3, 2))[..., [3, 1], :]
        # if scatter is not None:
        #     scatter = scatter.to(cfg.device)
        #     scatter[:, :, :2] = wrf_scaler.transform(scatter[:, :, :2], dims=2,
        #                                              means=wrf_scaler.means[:2],
        #                                              stds=wrf_scaler.stddevs[:2])

        optimizer.zero_grad()
        # print(train_data.data.shape)
        
        output = model(train_data)

        loss = criterion(train_data.data[:, :3], output.data, train_label.data) #, stations,
                        #  scatter, i, metadata['start_date'], wrf_scaler)
        # print(train_data.data[:, :3].dtype,  output.data.dtype, train_label.data.dtype)
        # print(loss.dtype, 'loss')  # Check loss dtype
        # print(next(model.parameters()).dtype, 'model')
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_var_epoch(model, criterion, wrf_scaler, era_scaler, dataloader, logger, cfg):
    # metadata = dataloader.dataset.metadata
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        valid_batch_count = 0
        for valid_data, valid_label, stations, scatter, i in tqdm(dataloader, disable=rank() != 0):
            if valid_data is None:
                continue
            valid_data = transform_packed_sequence_multiple(valid_data.to(cfg.device), [(torch.Tensor.type, (torch.float,), {}),
                                                                                        (wrf_scaler.transform, (), {'dims': 1})])
            valid_label = transform_packed_sequence_multiple(valid_label.to(cfg.device), [(torch.Tensor.type, (torch.float,), {}),
                                                                                          (era_scaler.transform, (), {'dims': 1})])
            # valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            # valid_label = torch.swapaxes(valid_label.type(torch.float).to(cfg.device), 0, 1)
            # valid_data = wrf_scaler.transform(valid_data, dims=2)
            # valid_label = era_scaler.transform(valid_label, dims=2)
            if stations is not None:
                stations = torch.permute(stations.type(torch.float).to(cfg.device), (1, 0, 3, 2))[..., [3, 1], :]
            if scatter is not None:
                scatter = scatter.to(cfg.device)
                scatter[:, :, :2] = wrf_scaler.transform(scatter[:, :, :2], dims=2,
                                                         means=wrf_scaler.means[:2],
                                                         stds=wrf_scaler.stddevs[:2])

            output = model(valid_data)

            # valid_data = valid_data[:, :, :3]
            loss = criterion(valid_data.data[:, :3],  output.data, valid_label.data, logger=logger) #, stations,
                            #  scatter, i, metadata['start_date'], wrf_scaler, logger)
            valid_loss += loss.item()
            valid_batch_count += 1

    return _reduce_validation_state(valid_loss, valid_batch_count, logger, cfg.device)
