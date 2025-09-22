import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, '../../')
from correction.config.config import cfg


def train(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, max_epochs):
    best_epoch = None
    try:
        for epoch in range(max_epochs):
            train_loss = train_epoch(train_dataloader, encoder_forecaster, criterion,
                                     optimizer, wrf_scaler, era_scaler)
            if logger:
                logger.train_loss.append(train_loss)
            print('train loss', train_loss)
            valid_loss = eval_epoch(encoder_forecaster, criterion, wrf_scaler, era_scaler, valid_dataloader, logger)
            print('valid_loss', valid_loss)
            if logger:
                logger.print_stat_readable(epoch)
            lr_scheduler.step()
            print(lr_scheduler.get_last_lr())
            if logger:
                best_epoch = logger.save_model(encoder_forecaster.state_dict(), epoch)
    except KeyboardInterrupt:
        pass
    logger.save_configuration() if logger else None
    return best_epoch, encoder_forecaster


def train_epoch(dataloader, model, criterion, optimizer, wrf_scaler, era_scaler):
    metadata = dataloader.dataset.metadata
    train_loss = 0
    model.train()
    t = 0
    for train_data, train_label, stations, scatter, i in (pbar := tqdm(dataloader)):
        train_data = torch.swapaxes(train_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        train_data = wrf_scaler.channel_transform(train_data, 2)

        train_label = torch.swapaxes(train_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        train_label = era_scaler.channel_transform(train_label, 2)

        stations = torch.permute(stations.type(torch.float).to(cfg.GLOBAL.DEVICE), (1, 0, 3, 2))[..., [3, 1], :]

        scatter = scatter.to(cfg.GLOBAL.DEVICE)
        scatter[:, :, :2] = wrf_scaler.channel_transform(scatter[:, :, :2], 2)

        optimizer.zero_grad()

        output = model(train_data)
        if cfg.run_config.use_spatiotemporal_encoding:
            train_data = train_data[:, :, :3]
        loss = criterion(train_data, output, train_label, stations,
                         scatter, i, metadata['start_date'], wrf_scaler)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_epoch(model, criterion, wrf_scaler, era_scaler, dataloader, logger):
    metadata = dataloader.dataset.metadata
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        for valid_data, valid_label, stations, scatter, i in tqdm(dataloader):
            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            if cfg.run_config.use_era_data:
                valid_label = torch.swapaxes(valid_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
                valid_label = era_scaler.channel_transform(valid_label, 2)
            else:
                valid_label = None
            if cfg.run_config.use_stations_data:
                stations = torch.permute(stations.type(torch.float).to(cfg.GLOBAL.DEVICE), (1, 0, 3, 2))[..., [3, 1], :]
            else:
                stations = None
            if scatter is not None:
                scatter = scatter.to(cfg.GLOBAL.DEVICE)
                scatter[:, :, :2] = wrf_scaler.channel_transform(scatter[:, :, :2], 2)
            valid_data = wrf_scaler.channel_transform(valid_data, 2)

            output = model(valid_data)
            if cfg.run_config.use_spatiotemporal_encoding:
                valid_data = valid_data[:, :, :3]
            loss = criterion(valid_data, output, valid_label, stations,
                             scatter, i, metadata['start_date'], wrf_scaler, logger)
            valid_loss += loss.item()

        valid_loss = valid_loss / len(dataloader)
    return valid_loss


def trial_model(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, max_epochs, trial=None):
    for epoch in range(max_epochs):
        train_loss = train_epoch(train_dataloader, encoder_forecaster, criterion,
                                 optimizer, wrf_scaler, era_scaler, None)

        print('train loss', train_loss)
        lr_scheduler.step()
        # trial.report(trial_loss, step=epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    print('Started epoch trial...')
    from correction.pipeline.test import test
    trial_loss = test(encoder_forecaster, criterion, wrf_scaler, era_scaler, valid_dataloader,
                      logger=logger, save_losses=False)
    print(trial_loss, 'trial acc')
    torch.save(encoder_forecaster.state_dict(), os.path.join(logger.model_save_dir, f'model_last.pth'))
    return trial_loss


def trial_epoch(model, criterion, wrf_scaler, era_scaler, dataloader):
    metadata = dataloader.dataset.metadata
    t = 0
    with torch.no_grad():
        model.eval()
        trial_loss = torch.zeros(4, device=cfg.GLOBAL.DEVICE)
        for valid_data, valid_label, stations, scatter, i in (pbar := tqdm(dataloader)):
            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            valid_label = torch.swapaxes(valid_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            valid_label = era_scaler.channel_transform(valid_label, 2)
            stations = torch.permute(stations.type(torch.float).to(cfg.GLOBAL.DEVICE), (1, 0, 3, 2))[..., [3, 1], :]
            scatter = scatter.to(cfg.GLOBAL.DEVICE)
            scatter[:, :, :2] = wrf_scaler.channel_transform(scatter[:, :, :2], 2)
            valid_data = wrf_scaler.channel_transform(valid_data, 2)

            output = model(valid_data)
            if cfg.run_config.use_spatiotemporal_encoding:
                valid_data = valid_data[:, :, :3]
            loss = criterion(valid_data, output, valid_label, stations,
                             scatter, i, metadata['start_date'], wrf_scaler, expanded_out=True)
            orig_loss = criterion(valid_data, valid_data, valid_label, stations,
                                  scatter, i, metadata['start_date'], wrf_scaler, expanded_out=True)
            loss = get_trial_losses(loss, orig_loss)
            pbar.set_description(f'{loss[0].sum().item()}')
            trial_loss += loss

        trial_loss = trial_loss / len(dataloader)
    return tuple(map(torch.Tensor.item, torch.split(trial_loss, 1)))


def get_trial_losses(loss, orig_loss):
    loss = torch.stack(loss)[[1, 3, 4]]
    orig_loss = torch.stack(orig_loss)[[1, 3, 4]]
    relative_loss = torch.zeros_like(loss)
    mask = (orig_loss != 0)
    relative_loss[mask] = (orig_loss[mask] - loss[mask]) / orig_loss[mask]
    loss = torch.cat([relative_loss.sum()[None], loss])
    return loss
