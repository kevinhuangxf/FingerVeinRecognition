import importlib


def set_param_groups(network, custom_params, optim_args):
    """set_param_groups for setting custom network parameters lr.

    Args:
        network(nn.Module): The network to be customized lr.
        custom_params(list | OmegaConf.ListConfig): List of dictionary for setting a
            parameters group.
        optim_args(dict | OmegaConf.DictConfig): The arguments for initializing optimizers.
    Returns:
        param_groups(list): List of parameters groups for initializing optimizers.
    """

    param_groups = []

    # set custom group
    for custom_param in custom_params:
        param_group = dict(
            name=custom_param['name'], lr=custom_param['lr_scale'] * optim_args.lr, params=[])
        param_groups.append(param_group)

    # set network group
    param_groups.append(dict(name='network', lr=optim_args.lr, params=[]))

    for k, v in network.named_parameters():
        for param_group in param_groups:
            # append params to custom group
            if param_group['name'] in k:
                param_group['params'].append(v)
                break
            # append params to network group
            elif param_group['name'] == 'network':
                param_group['params'].append(v)

    return param_groups


def get_optimizer(param_groups, optim_args):
    '''torch optimizer getter'''
    torch_optim_package = importlib.import_module('torch.optim')

    optim_type = optim_args.pop('type')

    if hasattr(optim_args, 'custom_params'):
        custom_params = optim_args.pop('custom_params')  # noqa: F841
        # param_groups = set_param_groups(network, custom_params, optim_args)
        optimizer = getattr(torch_optim_package, optim_type)(param_groups, **optim_args)
    else:
        optimizer = getattr(torch_optim_package, optim_type)(param_groups, **optim_args)

    return optimizer


def get_lr_scheduler(optimizer, sched_args):
    '''torch lr_scheduler getter'''
    torch_lr_package = importlib.import_module('torch.optim.lr_scheduler')

    sched_args = dict(sched_args)
    sched_type = sched_args.pop('type')
    lr_scheduler = getattr(torch_lr_package, sched_type)(optimizer, **sched_args)

    return lr_scheduler
