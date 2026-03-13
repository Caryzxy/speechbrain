#!/usr/bin/env python3

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.inference.speaker import SpeakerRecognition


class Classifier(nn.Module):
    def __init__(self, input_size=192, hidden_size=256, out_neurons=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_neurons),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.net(x)
        return self.log_softmax(x)


class FinetuneECAPABrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Let SpeechBrain's pretrained verifier handle its own feature pipeline
        embeddings = self.hparams.verifier.encode_batch(wavs, lens)

        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        predictions = self.modules.classifier(embeddings)
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        spkid, _ = batch.spk_id_encoded
        spkid = spkid.squeeze(-1).long()

        loss = F.nll_loss(predictions, spkid)

        self.loss_metric.append(batch.id, predictions, spkid, reduction="none")
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid)

        return loss

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(metric=F.nll_loss)
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

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


def dataio_prep(hparams):
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams["n_classes"])

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)

        # force mono 1D waveform
        if sig.dim() == 2:
            if sig.shape[1] <= 8:
                sig = sig.mean(dim=1)
            elif sig.shape[0] <= 8:
                sig = sig.mean(dim=0)

        return sig.squeeze()

    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        yield label_encoder.encode_label_torch(spk_id)

    dynamic_items = [audio_pipeline, label_pipeline]
    output_keys = ["id", "sig", "spk_id", "spk_id_encoded"]

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for split, csv_path in data_info.items():
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path,
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

    train_loader_opts = hparams["dataloader_options"].copy()
    valid_loader_opts = hparams["dataloader_options"].copy()
    test_loader_opts = hparams["dataloader_options"].copy()

    train_loader_opts["shuffle"] = True
    valid_loader_opts["shuffle"] = False
    test_loader_opts["shuffle"] = False

    return datasets, train_loader_opts, valid_loader_opts, test_loader_opts


def load_pretrained_verifier(savedir, device):
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": device},
    )
    return verifier


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets, train_loader_opts, valid_loader_opts, test_loader_opts = dataio_prep(hparams)

    device = getattr(run_opts, "device", "cpu")

    verifier = load_pretrained_verifier(
        hparams["pretrained_savedir"],
        device=device,
    )

    # Use the verifier's embedding model as the trainable encoder module
    embedding_model = verifier.mods.embedding_model

    if hparams.get("freeze_encoder", False):
        for p in embedding_model.parameters():
            p.requires_grad = False

    modules = {
        "embedding_model": embedding_model,
        "classifier": Classifier(
            input_size=hparams["emb_dim"],
            hidden_size=hparams["classifier_hidden"],
            out_neurons=hparams["n_classes"],
        ),
    }

    # store verifier in hparams so compute_forward can use encode_batch()
    hparams["verifier"] = verifier

    brain = FinetuneECAPABrain(
        modules=modules,
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hasattr(brain.checkpointer, "add_recoverable"):
        brain.checkpointer.add_recoverable("embedding_model", brain.modules.embedding_model)
        brain.checkpointer.add_recoverable("classifier", brain.modules.classifier)

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=train_loader_opts,
        valid_loader_kwargs=valid_loader_opts,
    )

    brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=test_loader_opts,
    )

    torch.save(
    brain.modules.embedding_model.state_dict(),
    os.path.join(hparams["output_folder"], "finetuned_embedding_model.pt"),
    )
    print("Saved fine-tuned encoder to:",
      os.path.join(hparams["output_folder"], "finetuned_embedding_model.pt"))