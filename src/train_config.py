class Config:
    
    def __init__(self):
        "Ugly hack to make the args object work without spending to much effort."
        self.local_rank = -1
        self.per_gpu_train_batch_size = 4
        self.gradient_accumulation_steps = 8
        self.n_gpu = 1
        self.max_steps = 2000
        self.weight_decay = 0
        self.learning_rate = 5e-5
        self.adam_epsilon = 2e-8
        self.warmup_steps = 250
        self.model_name_or_path = "bert"
        self.fp16 = False
        self.set_seed = "42"
        self.device=0
        self.model_type="roberta"
        self.no_cuda=False
        self.max_grad_norm=1.0
        self.logging_steps=1
        self.evaluate_during_training=False
        self.save_steps=2500
        self.output_dir="./"
        self.evaluate_dataset=""
