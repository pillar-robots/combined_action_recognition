from pathlib import Path

from PIL import Image
import torch
from .model import get_gazelle_model
from .utils import visualize_heatmap, visualize_all
import numpy as np
import cv2

# image = "/home/niki/pillar/842.jpg"


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "gazelle_dinov2_vitl14_inout.pt"


class GazeEstimator:
  def __init__(self, model_name="gazelle_dinov2_vitl14_inout", model_path=None):
    if model_path is None:
      model_path = str(DEFAULT_MODEL_PATH)
    self.model, self.transform = get_gazelle_model(model_name)
    self.model.load_gazelle_state_dict(torch.load(model_path, weights_only=True))
    self.model.eval()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    # if self.device == "cuda":
        # self.model = self.model.to(memory_format=torch.channels_last)
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    torch.backends.cudnn.benchmark = True  # speed up fixed shapes

  @torch.no_grad()
  def gaze_estimation(self, image, bboxes):
    # detect faces
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = image.size
    bbox_2 = []
    # print(bboxes)
    for i, bbox in enumerate(bboxes):
      x1, y1, x2, y2 = bbox # ["face"]
      bbox_2.append([x1, y1, x2, y2])

    bboxes = bbox_2

    # # prepare gazelle input
    img_tensor = self.transform(image).unsqueeze(0).to(self.device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]
    input = {
        "images": img_tensor, # [num_images, 3, 448, 448]
        "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
    }

    # if self.device == "cuda":
    #     input['images'] = input['images'].to(memory_format=torch.channels_last)
    # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
    output = self.model(input)

    # print(len(output['heatmap']), output['heatmap'][0].shape[0]) # [1, num_people, 64, 64]
    # raise
    img1_person1_heatmap = output['heatmap'][0][0] # [64, 64] heatmap
    if self.model.inout:
      img1_person1_inout = output['inout'][0][0] # gaze in frame score (if model supports inout prediction)

    # for i in range(len(bboxes)):
    #   viz = visualize_heatmap(image, output['heatmap'][0][i], norm_bboxes[0][i], inout_score=output['inout'][0][i] if output['inout'] is not None else None)
    #   viz = viz.convert("RGB")
    #   viz.save(f"gaze_output_{i}.jpg")


    viz2, gaze_target = visualize_all(image, output['heatmap'][0], norm_bboxes[0], output['inout'][0] if output['inout'] is not None else None, inout_thresh=0.5,
                        draw_overlay=False)
    # print(viz2.shape, len(gaze_target))
    # viz2 = viz2.convert("RGB")
    # viz2.save("gaze_output_all.jpg")

    return gaze_target





# def gaze_estimation(image, bboxes):
#   model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
#   model.load_gazelle_state_dict(torch.load("/home/niki/pillar/gazelle/gazelle_dinov2_vitl14_inout.pt", weights_only=True))
#   model.eval()

#   device = "cuda" if torch.cuda.is_available() else "cpu"
#   # CUDA_VISIBLE_DEVICES = 3
#   model.to(device)

#   # detect faces
#   image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#   width, height = image.size
#   bbox_2 = []
#   # print(bboxes)
#   for i, bbox in enumerate(bboxes):
#     x1, y1, x2, y2 = bbox # ["face"]
#     bbox_2.append([x1, y1, x2, y2])

#   bboxes = bbox_2

#   # # prepare gazelle input
#   img_tensor = transform(image).unsqueeze(0).to(device)
#   norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]
#   input = {
#       "images": img_tensor, # [num_images, 3, 448, 448]
#       "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
#   }
#   with torch.no_grad():
#       output = model(input)
#   img1_person1_heatmap = output['heatmap'][0][0] # [64, 64] heatmap
#   if model.inout:
#     img1_person1_inout = output['inout'][0][0] # gaze in frame score (if model supports inout prediction)

#   for i in range(len(bboxes)):
#     viz = visualize_heatmap(image, output['heatmap'][0][i], norm_bboxes[0][i], inout_score=output['inout'][0][i] if output['inout'] is not None else None)
#     viz = viz.convert("RGB")
#     viz.save(f"gaze_output_{i}.jpg")


#   viz2, gaze_target = visualize_all(image, output['heatmap'][0], norm_bboxes[0], output['inout'][0] if output['inout'] is not None else None, inout_thresh=0.5)
#   viz2 = viz2.convert("RGB")
#   viz2.save("gaze_output_all.jpg")

#   return gaze_target
