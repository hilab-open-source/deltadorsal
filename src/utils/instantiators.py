"""Hydra instantiation utilities for callbacks and loggers."""

import hydra
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig):
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig):
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger = []

    if not logger_cfg:
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
