import torch
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

device='cuda'

class BaseMetric():
  def __init__(self):
    self.count = torch.Tensor([0.]).to(device)
    self.score = torch.Tensor([0.]).to(device)
    self.std = torch.Tensor([0.]).to(device)
    self.score_list = torch.Tensor([]).to(device)

  def compute(self):
    raise NotImplementedError

  def update(self, ground_truth_image, image):
    s = self.compute(ground_truth_image, image)
    self.score = self.score * (self.count/(self.count+1.)) + s * (1./(self.count+1.))
    self.count = self.count + 1
    self.score_list = torch.cat([self.score_list, s.ravel()], axis=0)
    self.std = ((self.score_list - self.score) ** 2).mean()
    return self

  def reset(self):
    self.count = torch.Tensor([0.]).to(device)
    self.score = torch.Tensor([0.]).to(device)
    self.std = torch.Tensor([0.]).to(device)
    self.score_list = torch.Tensor([]).to(device)
    return self

  def print(self):
    print(f"{self.__class__.__name__} : {self.score.cpu().numpy()[0]} ± {self.std}")

class FID_metric():
  def __init__(self):
    self.metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    self.preprocess = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

  def score(self):
    return self.metric.compute()

  def update(self, image, real=False):
    image = self.preprocess(image).unsqueeze(0).to(device)
    self.metric.update(image, real=real)

  def reset(self):
    self.metric.reset()

  def print(self):
    print(f"{self.__class__.__name__} : {self.score()}")

class SSIM_metric(BaseMetric):
  def __init__(self):
    super().__init__()
    self.metric = StructuralSimilarityIndexMeasure().to(device)
    self.preprocess = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

  def compute(self, ground_truth_image, image):
    image = self.preprocess(image).unsqueeze(0).to(device)
    gt_im = self.preprocess(ground_truth_image).unsqueeze(0).to(device)
    score = torch.Tensor([self.metric(image, gt_im)])
    return score.to(device)

class PSNR_metric(BaseMetric):
  def __init__(self):
    super().__init__()
    self.metric = PeakSignalNoiseRatio().to(device)
    self.preprocess = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

  def compute(self, ground_truth_image, image):
    image = self.preprocess(image).unsqueeze(0).to(device)
    gt_im = self.preprocess(ground_truth_image).unsqueeze(0).to(device)
    score = torch.Tensor([self.metric(image, gt_im)])
    return score.to(device)

class LPIPS_metric(BaseMetric):
  def __init__(self):
    super().__init__()
    self.metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    self.preprocess = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

  def compute(self, ground_truth_image, image):
    image = self.preprocess(image).unsqueeze(0).to(device)
    gt_im = self.preprocess(ground_truth_image).unsqueeze(0).to(device)
    score = torch.Tensor([self.metric(image, gt_im)])
    return score.to(device)