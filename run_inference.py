import torch
import numpy as np
import cv2
import argparse
from path import Path
from models import DispResNet

parser = argparse.ArgumentParser(description='Inference script for DispNet to predict depth from a single image')
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument("--img-height", default=210, type=int, help="Image height")
parser.add_argument("--img-width", default=210, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--input-image", required=True, type=str, help="Path to input image")
parser.add_argument("--output-file", required=True, type=str, help="Path to output .npy file")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    args = parser.parse_args()

    # Load model
    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained, map_location=device)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    # Load and preprocess image
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = cv2.resize(img, (args.img_width, args.img_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))

    # Convert to tensor and normalize
    tensor_img = torch.from_numpy(img).unsqueeze(0).to(device)
    tensor_img = ((tensor_img / 255 - 0.45) / 0.225)

    # Predict depth
    output = disp_net(tensor_img)

    # Convert to depth
    depth = 1 / output.squeeze().cpu().numpy()

    # Save depth as .npy file
    np.save(args.output_file, depth)
    print(f"Depth prediction saved to {args.output_file}")

if __name__ == '__main__':
    main()