from itertools import combinations

import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn import preprocessing
from sklearn.metrics import roc_curve

from src.core.solvers.helper import get_lr_scheduler, get_optimizer


class SimCLRLitModel(LightningModule):

    def __init__(self, backbone, head, temperature=0.05, losses=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.save_hyperparameters(ignore=['backbone', 'head'])

    def forward(self, x):
        x = self.backbone(x)
        out = self.head(x)
        return out

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        (view_1, view_2), labels = batch
        features_1 = self.backbone(view_1)
        features_2 = self.backbone(view_2)
        head_features_1, outputs_1 = self.head(features_1)
        head_features_2, outputs_2 = self.head(features_2)

        feats = torch.cat([head_features_1, head_features_2], dim=0)

        # Encode all images
        # feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def on_validation_start(self):
        self.val_class_num = self.trainer.datamodule.data_val.class_num

    # def validation_step(self, batch, batch_idx):
    #     self.info_nce_loss(batch, mode="val")

    def validation_step(self, batch, batch_idx):
        # network step
        data, labels = batch
        features = self.backbone(data)
        head_features, outputs = self.head(features)

        outputs = outputs.cpu().numpy()
        head_features = head_features.cpu().numpy()
        labels = labels.cpu().numpy()

        return head_features, labels

    def validation_epoch_end(self, outputs):

        embeddings = [output[0] for output in outputs]
        targets = [output[1] for output in outputs]
        embeddings = np.vstack(embeddings)
        embeddings = preprocessing.normalize(embeddings)
        targets = np.concatenate(targets)

        emb_num = len(embeddings)

        # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
        scores = np.matmul(embeddings, embeddings.T)
        class_num = self.val_class_num
        samples_per_class = emb_num // class_num

        # define matching pairs
        intra_class_combinations = np.array(list(combinations(range(samples_per_class), 2)))
        match_pairs = [i * samples_per_class + intra_class_combinations for i in range(class_num)]
        match_pairs = np.concatenate(match_pairs, axis=0)
        scores_match = scores[match_pairs[:, 0], match_pairs[:, 1]]
        labels_match = np.ones(len(match_pairs))

        # define imposter pairs
        inter_class_combinations = np.array(list(combinations(range(class_num), 2)))
        imposter_pairs = [
            np.expand_dims(i * samples_per_class, axis=0) for i in inter_class_combinations
        ]
        imposter_pairs = np.concatenate(imposter_pairs, axis=0)
        scores_imposter = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
        labels_imposter = np.zeros(len(imposter_pairs))

        # merge matching pairs and imposter pairs and assign labels
        all_scores = np.concatenate((scores_match, scores_imposter))
        all_labels = np.concatenate((labels_match, labels_imposter))

        # compute roc, auc and eer
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)

        # return fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets

        fnr = 1 - tpr
        # find indices where EER, fpr100, fpr1000, fpr0, best acc occur
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        fpr100_idx = sum(fpr <= 0.01) - 1
        fpr1000_idx = sum(fpr <= 0.001) - 1
        fpr10000_idx = sum(fpr <= 0.0001) - 1
        fpr0_idx = sum(fpr <= 0.0) - 1

        # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        fpr100 = fnr[fpr100_idx]
        fpr1000 = fnr[fpr1000_idx]
        fpr10000 = fnr[fpr10000_idx]
        fpr0 = fnr[fpr0_idx]

        metrics = (eer, fpr100, fpr1000, fpr10000, fpr0)
        metrics_thred = (thresholds[eer_idx], thresholds[fpr100_idx], thresholds[fpr1000_idx],
                         thresholds[fpr10000_idx], thresholds[fpr0_idx])
        # print(
        # 'EER:%.2f%%, FRR@FAR=0.01: %.2f%%, FRR@FAR=0.001: %.2f%%, FRR@FAR=0.0001: %.2f%%, ' + \
        # 'FRR@FAR=0: %.2f%%, Aver: %.2f%%' % (eer * 100, fpr100 * 100, fpr1000 * 100, \
        #     fpr10000 * 100,fpr0 * 100, np.mean(metrics) * 100))

        print(f'EER: {eer * 100}')
        print(f'FRR@FAR=0: {fpr0 * 100}')
        print(f'FRR@FAR=0.01: {fpr100 * 100}')
        print(f'FRR@FAR=0.001: {fpr1000 * 100}')
        print(f'FRR@FAR=0.0001: {fpr10000 * 100}')
        print(f'Aver: {np.mean(metrics) * 100}')

        return metrics, metrics_thred

    def test_step(self, batch, batch_idx):
        data, labels = batch
        features = self.backbone(data)
        head_features, outputs = self.head(features)

        return outputs

    def configure_optimizers(self):
        optimizer = get_optimizer([
            dict(name='backbone', params=self.backbone.parameters()),
            dict(name='head', params=self.head.parameters()),
        ], self.hparams.optimizer)

        lr_scheduler = get_lr_scheduler(optimizer, self.hparams.lr_scheduler)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
