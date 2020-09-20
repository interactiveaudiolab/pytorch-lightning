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
    print("start", id(model))

    trainer = Trainer(
        default_root_dir=".",
        max_epochs=5,
        weights_summary=None,
        distributed_backend='ddp_spawn',
        gpus=[0, 1],
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    #assert result == 1
    print("end", id(trainer.get_model()))
    param = next(model.parameters())[0]
    print(param)

    seed_everything(1234)

    # test
    print(trainer.checkpoint_callback.best_model_path)
    result = trainer.test(datamodule=dm)
    param = next(trainer.get_model().parameters())[0]
    print(param)

    result = result[0]
    assert result['test_acc'] > 0.8


if __name__ == "__main__":
    foo()
