import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnx
import onnxruntime

# Define the super-resolution model
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    # Load the pretrained model
    model = SuperResolutionNet(upscale_factor=3)
    model.load_state_dict(torch.load('superres_epoch100-44c6958e.pth', map_location='cpu'))
    model.eval()

    # Load and preprocess the input image
    img = Image.open('bear_low.jpg').convert('YCbCr')
    img_y, img_cb, img_cr = img.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y).unsqueeze(0)

    # Export the model to ONNX format
    torch.onnx.export(model, img_y, "super_resolution.onnx", export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # Load the ONNX model
    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)

    # Run the model using ONNX Runtime
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Post-process the output
    img_out_y = ort_outs[0]
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # Merge channels and convert to RGB
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC)
        ]).convert("RGB")

    # Save the output image
    final_img.save("bear_high.jpg")
    print("Super-resolution image saved as 'bear_high.jpg'")

if __name__ == "__main__":
    main()
