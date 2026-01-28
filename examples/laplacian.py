import cv2
import torch
import fastcv

img = cv2.imread("../artifacts/binary.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img)
gray_tensor = fastcv.laplacian(img_tensor)
gray_np = gray_tensor.cpu().numpy()
cv2.imwrite("output_laplacian_cuda.jpg", gray_np)
laplacian_opencv = cv2.Laplacian(img, cv2.CV_16S, ksize=1)
laplacian_opencv = cv2.convertScaleAbs(laplacian_opencv)
cv2.imwrite("output_laplacian_cpu.jpg", laplacian_opencv)
print("saved laplacian image.")