import ast

import torch
import torchvision.ops.boxes as bops

from dataset import *
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches


torch.set_printoptions(10)
@torch.no_grad()
def iou_anchor(bbox1, bbox2):
	bbox1 = bbox1.clone().detach()
	bbox2 = bbox2.clone().detach()
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
	#convert to [x1, y1, x2, y2]
	#out = torch.diagonal(bops.box_iou(bbox1, bbox2)).unsqueeze(0)
	out = bops.box_iou(bbox1, bbox2) 
	#out = out.nan_to_num(1)
	return out

def write_to_file(file):
	a = get_dataset(64)
	f = open(file, "w")
	for it, (i, o) in enumerate(a):
		print(it)
		bboxes = o[o[:, :, :, 4] == 1][:, 2:4]
		tl_dims = torch.zeros_like(bboxes)
		bboxes = torch.concat([tl_dims, bboxes], dim=1).tolist()
		for b in bboxes:
			f.write(', '.join(map(str, b)) + "\n")

	f.close()
	print("Done writing bounding boxes to file")

def generate_anchors(iterations, k):
	groups = {i: [] for i in range(k)}
	file = "train_bboxes.txt"
	write_to_file(file)
	bboxes = []
	with open(file) as f:
		bboxes = torch.tensor([ast.literal_eval(f"[{line[:-1]}]") for line in f.readlines()])
	starting_groups = torch.randint(0, len(bboxes), (k, ))
	start = bboxes[starting_groups]
	bboxes = bboxes
	for i in range(iterations):
		dis = 1 - iou_anchor(bboxes, start)
		closest_cluster = dis.min(dim=1).indices.tolist()
		for ind, n in enumerate(closest_cluster):
			groups[n].append(bboxes[ind])
		for key, value in groups.items():
			v = torch.cat(value, dim=0).reshape(-1, 4)
			new_best = v.mean(dim=0)
			start[key] = new_best
			groups[key] = []
		print("Iteration:", i)
	return start

if __name__ == '__main__':
	anchors = generate_anchors(100, 5)
	print(anchors)






