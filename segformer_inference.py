from re import I
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
from torchvision import transforms
import requests
import io
from torch.nn.functional import softmax
import datetime
import torchvision.transforms.functional as TF
import pandas as pd
import tqdm
from PIL import ImageFile
import itertools
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 推論時の参考URL
# https://techblog.cccmkhd.co.jp/entry/2022/09/27/084306

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.id2label = {0: "car",
                         1: "other"}

        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        ## テストデータなのでmask画像はなし
        # mask_file_names = [f for f in os.listdir(self.root_dir) if '.gif' in f]

        self.images = sorted(image_file_names)
        self.masks = sorted(image_file_names)

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

        self.num_classes = 2
        self.label2id = {v:k for k,v in self.id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        # self.model = torch.load('checkpoint/epoch=58-step=26313.ckpt')
        # self.model.load_state_dict(checkpoint['state_dict'])

    # 推論ではmaskなしを入力とする必要がある（正解を知らないので）
    # https://stackoverflow.com/questions/71717638/typeerror-forward-missing-1-required-positional-argument-in-a-method
    def forward(self, images):
        outputs = self.model(pixel_values=images)
        return(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

def prediction_to_vis(prediction):
    #color_map = {
    #0:(0,0,0),
    #1:(255,0,0),
    #}
    # for submission
    color_map = {
    0:(0),
    1:(255),
    }

    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape)
    for i,c in color_map.items():
        vis[prediction == i] = color_map[i]
    return Image.fromarray(vis.astype(np.uint8)),vis.astype(int)

# def inference_image(i,image):
#     # preprocess = torchvision.transforms.Compose([
#     # transforms.ToTensor(),
#     # transforms.Resize((128,128)),
#     # ])
#     img_tensor = image
#     img_tensor = torch.unsqueeze(img_tensor, axis=0)
#     output = model_module(img_tensor)

#     logits = output[0]

#     upsampled_logits = nn.functional.interpolate(
#         logits,
#         size=(1280, 1918),
#         mode="bilinear",
#         align_corners=False
#     )

#     predicted = upsampled_logits.argmax(dim=1).cpu()
#     f, axarr = plt.subplots(1,2)
#     axarr[0].imshow(prediction_to_vis(predicted[0,:,:]))
#     axarr[1].imshow(image)
#     f.savefig("result_images/inference_result_{}.png".format(i))

def run_length_encode(mask): #function to change the predicted masks into wanted submission type according to instructions
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()#making it scalar
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

#encode
def self_rle_encode(img):
    # 3チャネルをグレースケールへ変換（サイズを1280,1918,3から1280,1918へ）
    img = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    h, w = img.shape
    print(img.shape)
    # img.save("img_mask_inference.png")
    print('width:  ', w)
    print('height: ', h)
    flattened_img = img.flatten() #０が黒　１が白
    # print("Flattened Image:", flattened_img)
    # print("Unique Values in Flattened Image:", np.unique(flattened_img))
    # print("len:",len(flattened_img))
    pixels = (img >= 1).flatten().astype(int)
    # print("len:",len(pixels))
    # print("Unique Values in pix", np.unique(pixels))
    pixels = np.concatenate([[0], pixels, [0]]) #配列の前後に０を追加している
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#decode
def self_rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        # print("lo {}".format(lo))
        # print("hi {}".format(hi))
    return img.reshape(shape)


# チェックポイントからモデルのパラメータをロード
checkpoint = torch.load('checkpoint/epoch=58-step=26313.ckpt')
print("checkpoint load is finished")

# インスタンスを作成、モデルのパラメータをセット
id2label = {0: "car", 1: "other"}
model_module = SegformerFinetuner(id2label)

model_module.load_state_dict(checkpoint['state_dict'])
print("model load is finished")

model_module.eval()
model_module.freeze()
print("model eval mode is finished")

# input_dir = "./split_data_small/test/"
input_dir = "./split_data/submission_data/"
image_file_names = [f for f in os.listdir(input_dir) if '.jpg' in f]
images_name = sorted(image_file_names)
# images_name = image_file_names
print(images_name)

## テストデータの特徴抽出

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.reduce_labels = False
feature_extractor.size = 128
batch_size = 1
test_dataset = SemanticSegmentationDataset(input_dir, feature_extractor)

test_images = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=8)

if not os.path.isdir("result_images"):
    os.mkdir("result_images")
count=0
for batch in test_images:
    all_predictions = []
    print("test_images{}".format(len(test_images)))
    print("batch {}".format(len(batch)))

    images, masks = batch['pixel_values'], batch['labels']
    outputs = model_module(images)

    logits = outputs[0]

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(1280,1918),
        mode="bilinear",
        align_corners=False
    )

    predicted = upsampled_logits.argmax(dim=1).cpu()
    # masks = masks.cpu().numpy()
    # f, axarr = plt.subplots(predicted.shape[0],2)
    for i in range(predicted.shape[0]):
        pred_image, pred_np = prediction_to_vis(predicted[i,:,:])
        # 入力画像と推論結果の比較
        # axarr[0].imshow(pred_image)
        # image = Image.open(os.path.join(input_dir, images_name[count]))
        # axarr[1].imshow(image)
        # save_file_name = "predicted_{}.png".format(count)
        # f.savefig("result_images/"+save_file_name)

        # 提出ファイルへの変換https://www.kaggle.com/code/vladivashchuk/pytorch-unet-with-submission
        pred_image.save("result_images/" + images_name[count])

        encoding = self_rle_encode(pred_np)

        # 正しくエンコードで来ているかを確認
        decording = self_rle_decode(encoding, shape=(1280,1918))
        decording = np.clip(decording * 255, a_min = 0, a_max = 255).astype(np.uint8)
        decording =Image.fromarray(decording)
        file_name = "result_images/" + images_name[count]+ "_decoding{}.png".format(count)
        decording.save(file_name)


        all_predictions.append([images_name[count], encoding])
        # all_predictions = itertools.chain.from_iterable(all_predictions)
        print(images_name[count])
        count+=1
        print(count)

        # Submission
        header = ['img', 'rle_mask']
        sub = pd.DataFrame(all_predictions,columns=header)
        if count == 0:
            sub.to_csv('submission.csv',index=False)
        else:
            sub.to_csv('submission.csv',index=False,mode="a",header=False)
    if count ==200:
        break

# Submission
# sub = pd.DataFrame(all_predictions)
# sub.columns = ['img', 'rle_mask']
# sub.to_csv('submission.csv', index=False)