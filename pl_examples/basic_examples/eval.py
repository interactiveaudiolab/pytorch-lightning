from pytorch_lightning import seed_everything, Trainer
from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule

import torch
import numpy


def foo():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    seed_everything(1234)

    dm = TrialMNISTDataModule(".")
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=".",
        max_epochs=5,
        weights_summary=None,
        distributed_backend='ddp_spawn',
        gpus=[0, 1],
        deterministic=True,
    )

    seed_everything(1234)
    trainer.fit(model, dm)
    param = next(model.parameters())[0][0]
    print(param)

    seed_everything(1234)
    result = trainer.test(datamodule=dm, ckpt_path=None)
    param = next(trainer.get_model().parameters())[0][0]
    print(param)

    result = result[0]
    assert result['test_acc'] > 0.8


if __name__ == "__main__":
    foo()
