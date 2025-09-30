import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw

def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)

def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]

# def visualize_heatmap(pil_image, heatmap, bbox=None):
#     if isinstance(heatmap, torch.Tensor):
#         heatmap = heatmap.detach().cpu().numpy()
#     heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
#     heatmap = plt.cm.jet(np.array(heatmap) / 255.)
#     heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
#     heatmap = Image.fromarray(heatmap).convert("RGBA")
#     heatmap.putalpha(128)
#     overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

#     if bbox is not None:
#         width, height = pil_image.size
#         xmin, ymin, xmax, ymax = bbox
#         draw = ImageDraw.Draw(overlay_image)
#         draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="green", width=3)
#     return overlay_image


# visualize predicted gaze heatmap for each person and gaze in/out of frame score

def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
          text = f"in-frame: {inout_score:.2f}"
          text_width = draw.textlength(text)
          text_height = int(height * 0.01)
          text_x = xmin * width
          text_y = ymax * height + text_height
          draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))
    return overlay_image

def stack_and_pad(tensor_list):
    max_size = max([t.shape[0] for t in tensor_list])
    padded_list = []
    for t in tensor_list:
        if t.shape[0] == max_size:
            padded_list.append(t)
        else:
            padded_list.append(torch.cat([t, torch.zeros(max_size - t.shape[0], *t.shape[1:])], dim=0))
    return torch.stack(padded_list)


# ego to val apo colab
def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5, draw_overlay=False):
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    if draw_overlay:
        draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    # create a gaze target to store the gaze target of each person
    gaze_target = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox

        if draw_overlay:
            color = colors[i % len(colors)]
            draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))

        if inout_scores is not None:
            inout_score = inout_scores[i]
            if draw_overlay:
                text = f"in-frame: {inout_score:.2f}"
                text_width = draw.textlength(text)
                text_height = int(height * 0.01)
                text_x = xmin * width
                text_y = ymax * height + text_height
                draw.text((text_x, text_y), text, fill=color, font=ImageFont.load_default(size=int(min(width, height) * 0.05)))
            

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            if draw_overlay:
                draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
                draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))

            gaze_target.append({"Person ID": i, "Gaze Target": (gaze_target_x, gaze_target_y)})
            # gaze_target.append((gaze_target_x, gaze_target_y))
        else:
            gaze_target.append({"Person ID": i, "Gaze Target": None})
            # gaze_target.append(None)



    return overlay_image, gaze_target
