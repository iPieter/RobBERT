from transformers import AdamW
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
from tqdm import tqdm, trange
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import json

from transformers import get_linear_schedule_with_warmup
import logging

logger = logging.getLogger(__name__)

class Train:
    def train(args, train_dataset, model, tokenizer, evaluate_fn=None):
        """ Train the model """

        model.train()

        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1
        args.device = device

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
            )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        #set_seed(args)  # Added here for reproductibility
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        if (
                            args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        epoch_iterator.set_postfix({**logs, **{"step": global_step}})

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break
                
            if evaluate_fn is not None:
                results = pd.DataFrame(evaluate_fn(args.evaluate_dataset, model))
                cm = ConfusionMatrix(actual_vector=results['true'].values, predict_vector=results['predicted'].values)
                logs = {}
                logs["eval_f1_macro"] = cm.F1_Macro
                logs["eval_acc_macro"] = cm.ACC_Macro
                logs["eval_acc_overall"] = cm.Overall_ACC
                logger.info("Results on eval: {}".format(logs))


        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step
