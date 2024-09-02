import torch

import cv2
import numpy as np
import pandas as pd
from path import Path
import argparse
from tqdm import tqdm

from models import DispResNet
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument("--img-height", default=210, type=int, help="Image height")
parser.add_argument("--img-width", default=210, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--image-list", default=None, type=str, help="File with list of image paths")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--position", required=True, type=str, help="Position identifier for output files")
parser.add_argument("--batch-size", default=1, type=int, help="Batch size for inference")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not (args.output_disp or args.output_depth):
        print('You must at least output one value!')
        return

    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained, map_location=device)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    output_dir = Path(args.output_dir) / args.position
    output_dir.makedirs_p()

    # Read image paths from the file
    image_paths_df = pd.read_csv(args.image_list, header=None, sep='\t')
    image_paths = image_paths_df[0].tolist()

    print('{} files to test'.format(len(image_paths)))

    # Create batches from image paths
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_files = image_paths[i:i + args.batch_size]

        # Load and preprocess images
        images = []
        for file in batch_files:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            h, w, _ = img.shape
            if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                img = cv2.resize(img, (args.img_width, args.img_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            images.append(img)

        tensor_imgs = torch.from_numpy(np.stack(images)).to(device)
        tensor_imgs = ((tensor_imgs / 255 - 0.45) / 0.225)

        # Predict with batch
        outputs = disp_net(tensor_imgs)

        # Save results
        for j, file in enumerate(batch_files):
            output = outputs[j]
            file_path, file_ext = Path(file).stem, Path(file).suffix
            file_name = file_path.replace('/', '_')

            if args.output_disp:
                disp = (255 * tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
                cv2.imwrite(str(output_dir / '{}_disp{}'.format(file_name, file_ext)), cv2.cvtColor(np.transpose(disp, (1, 2, 0)), cv2.COLOR_RGB2BGR))
            if args.output_depth:
                depth = 1 / output
                # Lưu file .npy
                np.save(str(output_dir / f'args.position_sfm.npy'), depth.cpu().numpy())
                depth = (255 * tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                cv2.imwrite(str(output_dir / '{}_depth{}'.format(file_name, file_ext)), cv2.cvtColor(np.transpose(depth, (1, 2, 0)), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()