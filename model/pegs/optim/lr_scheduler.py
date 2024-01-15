from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


SchedulerType = {
    "linear": "linear",
    "cosine": "cosine",
    "cosine_with_restarts": "cosine_with_restarts",
    "polynomial": "polynomial",
    "constant": "constant",
    "constant_with_warmup": "constant_with_warmup",
    "inverse_sqrt": "inverse_sqrt",
    "reduce_lr_on_plateau": "reduce_lr_on_plateau",
    "linear_with_warmup": get_linear_schedule_with_warmup,
    "cosine_with_warmup": get_cosine_schedule_with_warmup,
}
