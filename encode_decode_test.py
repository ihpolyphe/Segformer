from re import I
import numpy as np
import os
from PIL import Image,ImageFile
import tqdm
from matplotlib import pyplot as plt
import datetime
import pandas as pd
import tqdm
import itertools
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    h, w = img.shape
    # img.save("img_mask_inference.png")
    print('width:  ', w)
    print('height: ', h)
    flattened_img = img.flatten() #０が黒　１が白
    print("Flattened Image:", flattened_img)
    print("Unique Values in Flattened Image:", np.unique(flattened_img))
    print("len:",len(flattened_img))
    pixels = (img >= 1).flatten().astype(int)
    print("len:",len(pixels))
    print("Unique Values in pix", np.unique(pixels))
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
        print("lo {}".format(lo))
        print("hi {}".format(hi))
        img[lo:hi] = 1
    return img.reshape(shape)


input_dir = "train_masks.csv"
encode_data = pd.read_csv(input_dir)
print(len(encode_data))

# shape=[1280,1920]
for index, row in encode_data.iterrows():
    print(row["rle_mask"])
    decode_data_sample = self_rle_decode(row["rle_mask"], shape=(1280,1918))
    print(len(decode_data_sample.flatten()))
    decode_data_sample_image = np.clip(decode_data_sample * 255, a_min = 0, a_max = 255).astype(np.uint8)
    pil_image = Image.fromarray(decode_data_sample_image)
    #pil_image = Image.fromarray(decode_data_sample)
    pil_image.save("mask_decode.png")
    decode_data_sample_encode = self_rle_encode(decode_data_sample)
    decode_data_sample_encode_decode = self_rle_decode(decode_data_sample_encode, shape=(1280,1918))
    decode_data_sample_encode_decode = np.clip(decode_data_sample_encode_decode * 255, a_min = 0, a_max = 255).astype(np.uint8)
    pil_image = Image.fromarray(decode_data_sample_encode_decode)
    #pil_image = Image.fromarray(decode_data_sample)
    pil_image.save("mask_decode_encode_decode.png")
    break


# if not os.path.isdir("result_images"):
#     os.mkdir("result_images")
# count=0
# for batch in test_images:
#     all_predictions = []
#     print("test_images{}".format(len(test_images)))
#     print("batch {}".format(len(batch)))

#     images, masks = batch['pixel_values'], batch['labels']
#     outputs = model_module(images)

#     logits = outputs[0]

#     upsampled_logits = nn.functional.interpolate(
#         logits,
#         size=(1280,1920),
#         mode="bilinear",
#         align_corners=False
#     )

#     predicted = upsampled_logits.argmax(dim=1).cpu()
#     # masks = masks.cpu().numpy()
#     # f, axarr = plt.subplots(predicted.shape[0],2)
#     for i in range(predicted.shape[0]):
#         print("next")
#         pred_image, pred_np = prediction_to_vis(predicted[i,:,:])
#         # axarr[0].imshow(pred_image)
#         # image = Image.open(os.path.join(input_dir, images_name[count]))
#         # axarr[1].imshow(image)
#         # save_file_name = "predicted_{}.png".format(count)
#         # f.savefig("result_images/"+save_file_name)

#         # 提出ファイルへの変換https://www.kaggle.com/code/vladivashchuk/pytorch-unet-with-submission
#         pred_image.save("img_mask_inference.png")

#         encoding = self_rle_encode(pred_np)
#         size = [1280,1920]
#         decording = self_rle_decode(encoding, size)
#         decording =Image.fromarray(decording.astype(np.uint8))
#         decording.save("img_mask_inference_decording.png")


#         all_predictions.append([images_name[count], encoding])
#         # all_predictions = itertools.chain.from_iterable(all_predictions)
#         print(images_name[count])
#         count+=1
#         print(count)

#         # Submission
#         header = ['img', 'rle_mask']
#         sub = pd.DataFrame(all_predictions,columns=header)
#         if count == 0:
#             sub.to_csv('submission.csv',index=False)
#         else:
#             sub.to_csv('submission.csv',index=False,mode="a",header=False)

# # Submission
# # sub = pd.DataFrame(all_predictions)
# # sub.columns = ['img', 'rle_mask']
# # sub.to_csv('submission.csv', index=False)