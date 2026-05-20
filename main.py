import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from add_dark_channel import add_guide_channel
from models.networks import define_G


class desmoker():
    def __init__(self, model_path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is strictly required but not available. Please install PyTorch with CUDA support to use your RTX 4060 GPU.")
        self.device = torch.device("cuda")
        self.model = define_G()
        self.transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform_list)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def apply(self, img, dc_kernel_size=15, dc_filter_radius=15,
              dc_filter_eps=0.0001, bypass_dark_channel=False):
        """Run the defogging model on a BGR image.

        Args:
            img:                  BGR uint8 input (should be 256×256)
            dc_kernel_size:       Dark channel erosion kernel size
            dc_filter_radius:     Guided filter radius
            dc_filter_eps:        Guided filter regularization epsilon
            bypass_dark_channel:  If True, feed a zero dark channel

        Returns:
            result_rgb:  RGB uint8 output image
            dc_vis:      Grayscale dark channel visualization (H, W) uint8
        """
        im_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_ex, dc_vis = add_guide_channel(
            im_tmp,
            sz=dc_kernel_size,
            r=dc_filter_radius,
            eps=dc_filter_eps,
            bypass=bypass_dark_channel,
        )
        im_ex = self.transform(im_ex)
        with torch.no_grad():
            input_tensor = im_ex.unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
        return self.tensor2im(output.data), dc_vis
