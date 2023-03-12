import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as bops
from pprint import pprint
from time import perf_counter as pf

def iou(bbox1, bbox2):
	bbox1 = bbox1.clone()
	bbox2 = bbox2.clone()
	bbox1 += 1e-14
	bbox2 -= 1e-16
	bb1w = bbox1[:, 2].clone()
	bb1h = bbox1[:, 3].clone()
	bb2w = bbox2[:, 2].clone()
	bb2h = bbox2[:, 3].clone()
	bbox1[:, 2] = bbox1[:, 0] + bb1w / 2
	bbox1[:, 0] = bbox1[:, 0] - bb1w / 2
	bbox1[:, 3] = bbox1[:, 1] + bb1h / 2
	bbox1[:, 1] = bbox1[:, 1] - bb1h / 2

	bbox2[:, 2] = bbox2[:, 0] + bb2w / 2
	bbox2[:, 0] = bbox2[:, 0] - bb2w / 2
	bbox2[:, 3] = bbox2[:, 1] + bb2h / 2
	bbox2[:, 1] = bbox2[:, 1] - bb2h / 2

	#out = torch.diagonal(bops.box_iou(bbox1, bbox2)).unsqueeze(0)
	out = bops.box_iou(bbox1, bbox2)
	return out

import torch
from itertools import tee

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def generalized_diagonal(t):
    ratio = int(max(t.shape) / min(t.shape))
    indexes = ( (i, i*ratio) for i in range(min(t.shape)+1) )
    parts = [t[y0:y1, x0:x1] for (x0, y0), (x1, y1) in pairwise(indexes)]
    return torch.flatten(torch.stack(parts, dim=0))

class Yolov2Loss(nn.Module):
	def __init__(self, S=13, B=5, C=20, lambda_coord=5, lambda_noobj=0.5, lambda_class=1):
		super().__init__()
		self.S = S
		self.B = B
		self.C = C

		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.lambda_class = lambda_class
		self.mse = nn.MSELoss()
		self.cel = nn.CrossEntropyLoss()
		self.anchors = torch.tensor([[0.0000000000, 0.0000000000, 0.8010479212, 0.8268978596],
                               		 [0.0000000000, 0.0000000000, 0.6402753592, 0.4069810212],
                               		 [0.0000000000, 0.0000000000, 0.3496080637, 0.6819785237],
                               		 [0.0000000000, 0.0000000000, 0.0876114890, 0.1330402344],
                              		 [0.0000000000, 0.0000000000, 0.2124535590, 0.3432995677]])

	def forward(self, predictions, labels):
		anchors = self.anchors.to(predictions.device)

		object_inds = labels[..., 4] == 1
		_, cx, cy = torch.where(object_inds)

		obj_preds = predictions[object_inds]
		obj_labels = labels[object_inds]
		N_true = obj_labels.shape[0]

		noobj_preds = predictions[~object_inds]
		noobj_labels = labels[~object_inds]

		# make x and y with respect to whole image
		obj_preds[:, :, [0, 1, 4]] = torch.sigmoid(obj_preds[:, :, [0, 1, 4]])
		obj_preds[:, :, [2, 3]] = anchors[:, 2:] * torch.exp(obj_preds[:, :, [2, 3]])
		obj_preds[:, :, 0] = (obj_preds[:, :, 0] + cx.unsqueeze(1)) / self.S
		obj_preds[:, :, 1] = (obj_preds[:, :, 1] + cy.unsqueeze(1)) / self.S

		noobj_preds[:, :, 4] = torch.sigmoid(noobj_preds[:, :, 4])

		# target bounding box
		ious = iou(obj_preds[:, :, :4].view(-1, 4), obj_labels[:, :4])
		diagonal_ious = generalized_diagonal(ious).view(-1, 5)
		best_bbox_inds = diagonal_ious.argmax(dim=1)
		best_ious = diagonal_ious[range(N_true), best_bbox_inds]
		best_bboxes = obj_preds[range(N_true), best_bbox_inds]


		# Loss for coordinates

		# Loss for x and y

		loss_xy = self.mse(best_bboxes[:, :2], obj_labels[:, :2])

		# Loss for w and h

		loss_wh = self.mse(torch.sqrt(best_bboxes[:, 2:4]), torch.sqrt(obj_labels[:, 2:4]))

		# Loss for confidence

		# Loss with objects present
		
		loss_obj_conf = self.mse(obj_labels[:, 4], best_bboxes[:, 4])

		# Loss with no objects present

		mean_noobj_conf = noobj_preds[:, :, 4].mean(dim=1)
		loss_noobj_conf = self.mse(mean_noobj_conf, noobj_labels[:, 4])

		# Class Loss

		loss_class = self.cel(best_bboxes[:, 5:], obj_labels[:, 5:])

		# Final loss calcultion
		loss_coord = self.lambda_coord * (loss_xy + loss_wh)
		loss_confidence = loss_obj_conf + self.lambda_noobj * loss_noobj_conf
		loss_class = self.lambda_class * loss_class

		return loss_coord + loss_confidence + loss_class



