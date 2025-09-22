import os
import torch
from tqdm import tqdm
from correction.data.data_utils import transform_packed_sequence_multiple


def train(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, cfg):
    best_epoch = None
    try:
        for epoch in range(cfg.train.max_epochs):
            if cfg.run_config.variable_sequence_length:
                train_loss = train_var_epoch(train_dataloader, encoder_forecaster, criterion,
                                        optimizer, wrf_scaler, era_scaler, cfg)

                print('train loss', train_loss)
                valid_loss = eval_var_epoch(encoder_forecaster, criterion, wrf_scaler, era_scaler, valid_dataloader, logger,
                                        cfg)
            else:
                train_loss = train_epoch(train_dataloader, encoder_forecaster, criterion,
                                        optimizer, wrf_scaler, era_scaler, cfg)

                print('train loss', train_loss)
                valid_loss = eval_epoch(encoder_forecaster, criterion, wrf_scaler, era_scaler, valid_dataloader, logger,
                                        cfg)
            print('valid_loss', valid_loss)                
            lr_scheduler.step()
            print(lr_scheduler.get_last_lr())
            if logger:
                logger.train_loss.append(train_loss)
                logger.print_stat_readable(epoch)
                best_epoch = logger.save_model(encoder_forecaster.state_dict(), epoch)
                if epoch - best_epoch > 5:
                    break
    except KeyboardInterrupt:
        pass
    logger.save_configuration() if logger else None
    return best_epoch, encoder_forecaster


def train_epoch(dataloader, model, criterion, optimizer, wrf_scaler, era_scaler, cfg):
    train_loss = 0
    model.train()
    t = 0
    for train_data, train_label, stations, scatter, i in (pbar := tqdm(dataloader)):
        if train_data is None:
            continue
        train_data = torch.swapaxes(train_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
        train_data = wrf_scaler.transform(train_data, dims=2)
        train_label = torch.swapaxes(train_label.type(torch.float).to(cfg.device), 0, 1)
        train_label = era_scaler.transform(train_label, dims=2)
        if stations is not None:
            stations = torch.permute(stations.type(torch.float).to(cfg.device), (1, 0, 3, 2))[..., [3, 1], :]
        if scatter is not None:
            scatter = scatter.to(cfg.device)
            scatter[:, :, :2] = wrf_scaler.transform(scatter[:, :, :2], dims=2,
                                                     means=wrf_scaler.means[:2],
                                                     stds=wrf_scaler.stddevs[:2])

        optimizer.zero_grad()

        output = model(train_data)

        train_data = train_data[:, :, :3]
        loss = criterion(train_data, output, train_label, stations,
                         scatter, i, None, wrf_scaler)
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
        for valid_data, valid_label, stations, scatter, i in tqdm(dataloader):
            if valid_data is None:
                continue
            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            valid_label = torch.swapaxes(valid_label.type(torch.float).to(cfg.device), 0, 1)
            valid_data = wrf_scaler.transform(valid_data, dims=2)
            valid_label = era_scaler.transform(valid_label, dims=2)
            if stations is not None:
                stations = torch.permute(stations.type(torch.float).to(cfg.device), (1, 0, 3, 2))[..., [3, 1], :]
            if scatter is not None:
                scatter = scatter.to(cfg.device)
                scatter[:, :, :2] = wrf_scaler.transform(scatter[:, :, :2], dims=2,
                                                         means=wrf_scaler.means[:2],
                                                         stds=wrf_scaler.stddevs[:2])

            output = model(valid_data)

            valid_data = valid_data[:, :, :3]
            loss = criterion(valid_data, output, valid_label, stations,
                             scatter, i, None, wrf_scaler, logger=logger)
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

    print('Started epoch trial...')
    from correction.pipeline.test import test
    trial_loss = test(encoder_forecaster, criterion, wrf_scaler, era_scaler, valid_dataloader,
                      logger=logger, save_losses=False)
    print(trial_loss, 'trial acc')
    torch.save(encoder_forecaster.state_dict(), os.path.join(logger.model_save_dir, f'model_last.pth'))
    return trial_loss


def get_trial_losses(loss, orig_loss):
    loss = torch.stack(loss)[[1, 3, 4]]
    orig_loss = torch.stack(orig_loss)[[1, 3, 4]]
    relative_loss = torch.zeros_like(loss)
    mask = (orig_loss != 0)
    relative_loss[mask] = (orig_loss[mask] - loss[mask]) / orig_loss[mask]
    loss = torch.cat([relative_loss.sum()[None], loss])
    return loss


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
        for valid_data, valid_label, stations, scatter, i in tqdm(dataloader):
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

        valid_loss = valid_loss / len(dataloader)
    return valid_loss