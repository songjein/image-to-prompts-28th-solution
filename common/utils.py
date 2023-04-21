import inspect
import logging
import os
import sys
from datetime import datetime
from logging import Logger
from typing import Optional, TextIO

import numpy as np
import torch
import tqdm

__LOG_FORMAT = "Rank({rank}) | %(asctime)s | %(levelname)8s | %(message)s"
TQDM_FORMAT = (
    "{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


class TQDMHandler(logging.Handler):
    """tqdm 사용시 ``logging`` 모듈과 같이 사용가능한 handler입니다.
    .. code-block:: python
        import time
        import logging
        import sys
        import tqdm
        from gpt2.utils import TQDMHandler
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = TQDMHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)
        for i in tqdm.tqdm(range(1000)):
            logger.info("안녕")
            time.sleep(0.1)
    """

    def __init__(self, stream: Optional[TextIO] = None):
        super().__init__()
        if stream is None:
            stream = sys.stdout

        self.stream = stream

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
            tqdm.tqdm.write(message, self.stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def create_logger(
    name: Optional[str] = None,
    log_output_path: Optional[str] = None,
    level=logging.INFO,
    rank: int = 0,
) -> Logger:
    """
    logger를 생성합니다.
    :param name : logger의 이름
    :param log_output_path : log가 남을 디렉토리 경로
    :param level: logging level (기본은 INFO)
    :param rank: 멀티 프로세스 환경에서 각 프로세스의 글로벌 랭크
    """

    logger = logging.getLogger(name or __get_caller_module_name())
    logger.setLevel(level)
    formatter = logging.Formatter(__LOG_FORMAT.format(rank=rank))

    if rank == 0 and log_output_path:
        filename = os.path.join(
            log_output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f.log")
        )
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    tqdm_handler = TQDMHandler(sys.stdout)
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)

    logger.propagate = False

    return logger


class LayerwiseDecayAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        base_lr,
        min_lr=1e-6,
        backbone_weight_decay=1e-3,
        head_weight_decay=1e-5,
        head_lr_factor=10.0,
    ):
        params = []

        layers = [model.encoder.model.embeddings] + list(
            model.encoder.model.encoder.layers
        )
        learning_rates = np.linspace(base_lr, min_lr, len(layers))
        for idx, layer in enumerate(reversed(layers)):
            lr = learning_rates[idx]
            params += [
                {
                    "params": layer.parameters(),
                    "lr": lr,
                    "weight_decay": backbone_weight_decay,
                }
            ]

        head_lr = base_lr * head_lr_factor
        layers = [model.encoder.projection_features] + [model.encoder.clip_projection]
        for layer in layers:
            params += [
                {
                    "params": layer.parameters(),
                    "lr": head_lr,
                    "weight_decay": head_weight_decay,
                }
            ]

        super(LayerwiseDecayAdamW, self).__init__(
            params, defaults=dict(weight_decay=backbone_weight_decay)
        )
        self._optimizer = torch.optim.AdamW(self.param_groups)

    def step(self, closure=None):
        return self._optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        return self._optimizer.zero_grad(set_to_none=set_to_none)
