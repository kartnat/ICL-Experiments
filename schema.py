from funcy import merge
from quinine import (
    default,
    nullable,
    required,
    stdict,
    tboolean,
    tfloat,
    tinteger,
    tstring,
)


model_schema = {
    "n_positions": merge(tinteger, required),
    "n_dims": merge(tinteger, required),
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "loss": merge(tstring, nullable, default("mse")),
}

training_schema = {
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "exp_name": merge(tstring, nullable, default(None)),
    "run_altpaper_adam": merge(tboolean, default(False)),
}
