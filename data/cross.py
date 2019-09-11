import torch
window_res = 700
edge = round(window_res / 6)

cx = torch.Tensor([186.5, 106, 345.5, 508.5, 428, 268.5, 187, 345, 508, 423, 268, 111]).unsqueeze(1)
cy = torch.Tensor([163.5, 300.5, 166, 163.5, 305, 303.5, 438, 444, 439, 578, 577.5, 580]).unsqueeze(1)

rotation = torch.Tensor([170, 185, 175, 185, 180, 177, 193, 182, 187, 175, 165, 172]).unsqueeze(1)
scale = torch.Tensor([0.24, 0.22, 0.23, 0.26, 0.24, 0.28, 0.29, 0.28, 0.24, 0.25, 0.25, 0.21]).unsqueeze(1)

cx = ((cx - cx.min())/(cx.max()-cx.min()))*(window_res-2*edge) + edge
cy = ((cy - cy.min())/(cy.max()-cy.min()))*(window_res-2*edge) + edge

input_points = torch.cat((cx, cy, rotation, scale),1)
torch.save(input_points,'cross.pt')