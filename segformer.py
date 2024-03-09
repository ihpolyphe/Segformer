from torch.utils.data import Dataset, DataLoader
from transformers import SegformerFeatureExtractor
import numpy as np
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from datasets import load_metric
import os
import torch
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from PIL import Image
import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
# https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/
from torchmetrics import F1Score
import torch.multiprocessing as multiprocessing
import datetime
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.id2label = {0: "car",
                         1: "other"}

        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir) if '.gif' in f]

        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.root_dir, self.masks[idx]))
        # print("encoded_inputs start. resize maybe takes time")
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        # print("encoded_inputs finishued")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs


class SegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024", # 解像度1280*1918なのでそれに見合うよう大きい解像度の学習モデルを選択、またモデルサイズも更新
            # https://github.com/NVlabs/SegFormer
            # "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")
        dt_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.lg_f = str(dt_now) + '_f1.csv'

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)

    def training_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']
        masks = masks.to(device)

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1).to(device)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        f1_metric = F1Score(num_classes=2,mdmc_average='global').to(device)
        f1_metric = f1_metric(predicted,masks).to(device)
        f1_list = ["train f1 \n", str(f1_metric)]
        with open(self.lg_f, mode='a') as f:
            f.write('\n'.join(f1_list))
        # print("{}".format(self.lg_f))
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )

            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

            for k,v in metrics.items():
                self.log(k,v)

            return(metrics)
        else:
            return({'loss': loss})

    def validation_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']
        masks = masks.to(device)

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1).to(device)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        f1_metric = F1Score(num_classes=2,mdmc_average='global').to(device)
        f1_metric = f1_metric(predicted,masks).to(device)
        f1_list = ["validation f1", str(f1_metric)]
        with open(self.lg_f, mode='a') as f:
            f.write('\n'.join(f1_list))
        # print("{}".format(self.lg_f))
        # print("validation data f1 score is  {}".format(f1_metric))

        return({'val_loss': loss})

    def validation_epoch_end(self, outputs):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes,
              ignore_index=255,
              reduce_labels=False,
          )

        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        # val_f1 = metrics["f1"]

        # metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy, "f1":val_f1}
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)

        return metrics

    def test_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']
        masks = masks.to(device)

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1).to(device)

        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        f1_metric = F1Score(num_classes=2,mdmc_average='global').to(device)
        f1_metric = f1_metric(predicted,masks).to(device)
        f1_list = ["test f1", str(f1_metric)]
        with open(self.lg_f, mode='a') as f:
            f.write('\n'.join(f1_list))
        # print("{}".format(self.lg_f))
        # print("test data f1 score is  {}".format(f1_metric))

        return({'test_loss': loss})

    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes,
              ignore_index=255,
              reduce_labels=False,
          )

        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]
        # test_f1 = metrics["f1"]

        # metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy,"test_f1":test_f1}
        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}

        for k,v in metrics.items():
            self.log(k,v)

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.reduce_labels = False
feature_extractor.size = 128

train_dataset = SemanticSegmentationDataset("./split_data/train/", feature_extractor)
val_dataset = SemanticSegmentationDataset("./split_data/val/", feature_extractor)
test_dataset = SemanticSegmentationDataset("./split_data/test/", feature_extractor)


batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, prefetch_factor=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=8)

segformer_finetuner = SegformerFinetuner(
    train_dataset.id2label,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    metrics_interval=10,
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="min",
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

trainer = pl.Trainer(
    gpus=1,
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=500,
    val_check_interval=len(train_dataloader),
)

trainer.fit(segformer_finetuner)
checkpoint_callback.best_model_path
res = trainer.test(ckpt_path="best")

color_map = {
    0:(0,0,0),
    1:(255,0,0),
}

def prediction_to_vis(prediction):
    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape)
    for i,c in color_map.items():
        vis[prediction == i] = color_map[i]
    return Image.fromarray(vis.astype(np.uint8))

for batch in test_dataloader:
    images, masks = batch['pixel_values'], batch['labels']
    outputs = segformer_finetuner.model(images, masks)

    loss, logits = outputs[0], outputs[1]

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    predicted = upsampled_logits.argmax(dim=1).cpu()
    masks = masks.cpu().numpy()

f, axarr = plt.subplots(predicted.shape[0],2)
for i in range(predicted.shape[0]):
    axarr[i,0].imshow(prediction_to_vis(predicted[i,:,:]))
    axarr[i,1].imshow(prediction_to_vis(masks[i,:,:]))
    f.savefig("predicted.png")