from pytorch_lightning import seed_everything, Trainer
from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule

import torch
import numpy


def foo():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    seed_everything(1234)
    print("1")
    print(torch.rand(1,2))
    print(numpy.random.uniform(0, 1, 3))

    dm = TrialMNISTDataModule(".")

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=".",
        max_epochs=5,
        weights_summary=None,
        distributed_backend='ddp_spawn',
        gpus=[0, 1],
        deterministic=True,
        replace_sampler_ddp=False,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1

    print("2")
    print(torch.rand(1, 2))
    print(numpy.random.uniform(0, 1, 3))

    seed_everything(1234)

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


if __name__ == "__main__":
    foo()
