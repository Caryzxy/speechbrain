#!/usr/bin/env python3
"""
Train a speaker identification model on custom CSV manifests (e.g. RAVDESS).

This version is adapted from the SpeechBrain speaker_id template, but:
- removes MiniLibriSpeech-specific preparation
- removes noise augmentation / HDF5 cached features
- loads train/valid/test from CSV manifests
- uses a simple custom speaker encoder + classifier
"""

import os
import sys

import torch
import torch.nn as nn
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb


class SimpleSpeakerEncoder(nn.Module):
    def __init__(
        self,
        in_channels=23,
        hidden_channels=128,
        emb_dim=512,
    ):
        super().__init__()

        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
        )

        self.linear = nn.Linear(hidden_channels * 2, emb_dim)

    def forward(self, x, lengths=None):
        """
        x: [batch, time, feat_dim]
        returns: [batch, emb_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, T, F], got shape {x.shape}")

        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        x = self.tdnn(x)

        # statistics pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)

        stats = torch.cat([mean, std], dim=1)
        emb = self.linear(stats)
        return emb


class Classifier(nn.Module):
    def __init__(
        self,
        input_size,
        lin_blocks=1,
        lin_neurons=256,
        out_neurons=24,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, lin_neurons))
        layers.append(nn.BatchNorm1d(lin_neurons))
        layers.append(nn.LeakyReLU())

        for _ in range(lin_blocks - 1):
            layers.append(nn.Linear(lin_neurons, lin_neurons))
            layers.append(nn.BatchNorm1d(lin_neurons))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(lin_neurons, out_neurons))

        self.net = nn.Sequential(*layers)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        x: [batch, emb_dim] or [batch, 1, emb_dim]
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.net(x)
        return self.log_softmax(x)


class SpkIdBrain(sb.Brain):
    """Class that manages the training loop."""

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, lens = batch.sig

        feats = compute_features(
            wavs,
            lens,
            self.modules.compute_features,
            self.modules.mean_var_norm,
        )

        embeddings = self.modules.embedding_model(feats, lens)
        predictions = self.modules.classifier(embeddings)

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        spkid, _ = batch.spk_id_encoded
        spkid = spkid.squeeze(-1).long()

        loss = torch.nn.functional.nll_loss(predictions, spkid)

        self.loss_metric.append(
            batch.id, predictions, spkid, reduction="batch"
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid)

        return loss

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def compute_features(signal, length, compute_features_module, mean_var_norm):
    feats = compute_features_module(signal)
    return mean_var_norm(feats, length)


def dataio_prep(hparams):
    """
    Prepare datasets from CSV manifests.
    Expected CSV columns include:
    - ID
    - wav
    - spk_id
    """

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams["n_classes"])

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)

        # Ensure waveform is always 1D: [time]
        if sig.dim() == 2:
            # If shape is [time, channels], average channels
            if sig.shape[1] <= 8:
                sig = sig.mean(dim=1)
            # If shape is [channels, time], average channels
            elif sig.shape[0] <= 8:
                sig = sig.mean(dim=0)

        return sig.squeeze()

    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_label_torch(spk_id)
        yield spk_id_encoded

    dynamic_items = [audio_pipeline, label_pipeline]
    output_keys = ["id", "sig", "spk_id", "spk_id_encoded"]

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    # Usually shuffle train only
    train_loader_opts = hparams["dataloader_options"].copy()
    valid_loader_opts = hparams["dataloader_options"].copy()
    test_loader_opts = hparams["dataloader_options"].copy()

    train_loader_opts["shuffle"] = True
    valid_loader_opts["shuffle"] = False
    test_loader_opts["shuffle"] = False

    for split in data_info:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=data_info[split],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="spk_id",
    )

    return datasets, train_loader_opts, valid_loader_opts, test_loader_opts


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets, train_loader_opts, valid_loader_opts, test_loader_opts = dataio_prep(
        hparams
    )

    spk_id_brain = SpkIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    spk_id_brain.fit(
        epoch_counter=spk_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=train_loader_opts,
        valid_loader_kwargs=valid_loader_opts,
    )

    spk_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=test_loader_opts,
    )