import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import os, glob, platform, datetime, random
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(dir, phase, test_scene=None):
    images_paths = glob.glob(os.path.join(dir, 'clean', '*', '*.png'))
    albedo_paths = images_paths[:]
    shading_paths = images_paths[:]
    pathes = []
    for img_path in images_paths:
        sp = img_path.split('/'); 
        if phase == 'train':
            if sp[-2] == test_scene: continue
        else:
            if sp[-2] != test_scene: continue
            
        sp[-3] = 'albedo'; 
        sp = ['/'] + sp; 
        albedo_path = os.path.join(*sp)
        
        sp = img_path.split('/'); 
        sp[-3] = 'shading'; 
        sp[-1] = sp[-1].replace('frame', 'out')
        sp = ['/'] + sp; 
        shading_path = os.path.join(*sp)
        
        pathes.append((img_path, albedo_path, shading_path))
    return pathes

class MyImageFolder(data_utils.Dataset):
    def __init__(self, root, phase='train', transform=None, target_transform=None, random_crop=True, loader=default_loader, img_extentions=None, test_scene=None, image_h=None, image_w=None):
        imgs = make_dataset(root, phase, test_scene=test_scene)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(img_extentions)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.random_crop = random_crop
        self.image_h = image_h
        self.image_w = image_w
        
    def __getitem__(self, index):
        img_path, albedo_path, shading_path = self.imgs[index]
        
        img = self.loader(img_path)
        albedo = self.loader(albedo_path)
        shading = self.loader(shading_path)
        
        if self.random_crop == True:
            i, j, h, w = self.get_params(img, (int(self.image_h), int(self.image_w)))
            img = img.crop((j, i, j + w, i + h))
            albedo = albedo.crop((j, i, j + w, i + h))
            shading = shading.crop((j, i, j + w, i + h))
#         print(img.size)
#         print((i, j, h, w))
        
        if self.transform is not None: img = self.transform(img)
        if self.transform is not None: albedo = self.transform(albedo)
        if self.transform is not None: shading = self.transform(shading)
        
        scene = img_path.split('/')[-2]
        return img, albedo, shading, scene, img_path
    
    def __len__(self):
        return len(self.imgs)
    
    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    
