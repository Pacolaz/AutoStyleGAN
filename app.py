import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from types import SimpleNamespace
import random
from torchvision.utils import save_image
import gradio as gr
import numpy as np
import io
import tempfile  # Importar tempfile
import math

# Asegúrate de que las funciones necesarias estén definidas (si no lo están ya)
def resize(img, size):
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)

def denormalize(x):
    return (x + 1) / 2  # Valores en [0, 1]

# Definición de las clases de los modelos (Generator, StyleEncoder, MappingNetwork, ResBlk, AdaIN, AdainResBlk)
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False):
        super().__init__()
        self.normalize = normalize
        self.downsample = downsample
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1),
            nn.InstanceNorm2d(dim_out, affine=True) if normalize else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, 1),
            nn.InstanceNorm2d(dim_out, affine=True) if normalize else nn.Identity()
        )
        self.downsample_layer = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.skip = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.main(x)
        out = self.downsample_layer(out)
        skip = self.skip(x)
        skip = self.downsample_layer(skip)
        return (out + skip) / math.sqrt(2)

class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim):
        super(AdaIN, self).__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return (1 + gamma) * x + beta

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.w_hpf = w_hpf
        self.norm1 = AdaIN(dim_in, style_dim)
        self.norm2 = AdaIN(dim_out, style_dim)
        self.actv = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        if dim_in != dim_out:
            self.skip = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        else:
            self.skip = nn.Identity()

    def forward(self, x, s):
        x_orig = x
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x_orig = F.interpolate(x_orig, scale_factor=2, mode='nearest')
        h = self.norm1(x, s)
        h = self.actv(h)
        h = self.conv1(h)
        h = self.norm2(h, s)
        h = self.actv(h)
        h = self.conv2(h)
        skip = self.skip(x_orig)
        out = (h + skip) / math.sqrt(2)
        return out

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=True, downsample=True)]
            dim_in = dim_out
        self.encode = nn.Sequential(*blocks)
        self.decode = nn.ModuleList()
        for _ in range(repeat_num):
            dim_out = dim_in // 2
            self.decode += [AdainResBlk(dim_in, dim_out, style_dim, upsample=True)]
            dim_in = dim_out
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, 3, 1, 1, 0)
        )

    def forward(self, x, s):
        x = self.encode(x)
        for block in self.decode:
            x = block(x, s)
        out = self.to_rgb(x)
        return out

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2, hidden_dim=512):
        super(MappingNetwork, self).__init__()
        layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        ]
        for _ in range(3):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ]
        self.shared = nn.Sequential(*layers)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(hidden_dim, style_dim))

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)
        idx = torch.arange(y.size(0)).to(y.device)
        s = out[idx, y]
        return s

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=True, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_in, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = F.adaptive_avg_pool2d(h, (1,1))
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.arange(y.size(0)).to(y.device)
        s = out[idx, y]
        return s

# Clase para cargar imagenes
class ImageFolder(Dataset):
    def __init__(self, root, transform, mode, which='source'):
        self.transform = transform
        self.paths = []
        domains = sorted(os.listdir(root))
        for domain in domains:
            if os.path.isdir(os.path.join(root, domain)):
                files = os.listdir(os.path.join(root, domain))
                files = [os.path.join(root, domain, f) for f in files]
                self.paths += [(f, domains.index(domain)) for f in files]
        if mode == 'train' and which == 'reference':
            random.shuffle(self.paths)

    def __getitem__(self, index):
        path, label = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

    def __len__(self):
        return len(self.paths)

# Funciones para obtener los data loaders
def get_transform(img_size, mode='train', prob=0.5):
    transform = []
    transform.append(transforms.Resize((img_size, img_size)))
    if mode == 'train':
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomApply([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0))
        ], p=prob))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transform)

def get_train_loader(root, which='source', img_size=256, batch_size=8, prob=0.5, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(root=root, transform=transform, mode=which)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader

def get_test_loader(root, img_size=256, batch_size=8, shuffle=False, num_workers=4, mode='reference'):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(root=root, transform=transform, mode=mode)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return loader

# Clase Solver (adaptada para la inferencia)
class Solver(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Definir los modelos
        self.G = Generator(args.img_size, args.style_dim).to(self.device)
        self.M = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains).to(self.device)
        self.S = StyleEncoder(args.img_size, args.style_dim, args.num_domains).to(self.device)

    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.G.load_state_dict(checkpoint['generator'])
            self.M.load_state_dict(checkpoint['mapping_network'])
            self.S.load_state_dict(checkpoint['style_encoder'])
            print(f"Checkpoint cargado exitosamente desde {checkpoint_path}.")
        except FileNotFoundError:
            print(f"Error: No se encontró el checkpoint en {checkpoint_path}.")
            raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")
        except Exception as e:
            print(f"Error al cargar el checkpoint: {e}.")
            raise Exception(f"Error al cargar el checkpoint: {e}")

    def transfer_style(self, source_image, reference_image):
        # Asegúrate de que los modelos estén en modo de evaluación
        self.G.eval()
        self.S.eval()

        with torch.no_grad():
            # Preprocesar las imágenes de entrada
            transform = transforms.Compose([
                transforms.Resize((self.args.img_size, self.args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Convertir a PIL image antes de la transformación
            source_image_pil = Image.fromarray(source_image)
            reference_image_pil = Image.fromarray(reference_image)

            source_image = transform(source_image_pil).unsqueeze(0).to(self.device)
            reference_image = transform(reference_image_pil).unsqueeze(0).to(self.device)

            # Codificar el estilo de la imagen de referencia
            s_ref = self.S(reference_image, torch.tensor([0]).to(self.device))

            # Generar la imagen con el estilo transferido
            generated_image = self.G(source_image, s_ref)

            # Denormalizar la imagen para mostrarla en la interfaz
            generated_image = denormalize(generated_image.squeeze(0)).cpu()
            return (generated_image * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy() # Convertir a NumPy y a rango válido

# Función principal para la inferencia
def main(source_image, reference_image, checkpoint_path, args):
    if source_image is None or reference_image is None:
        raise gr.Error("Por favor, proporciona ambas imágenes (fuente y referencia).")

    # Crear el solver
    solver = Solver(args)
    # Cargar el checkpoint
    solver.load_checkpoint(checkpoint_path)

    # Realizar la transferencia de estilo
    generated_image = solver.transfer_style(source_image, reference_image)
    return generated_image

def gradio_interface():
    # Definir los argumentos (ajustados para la inferencia)
    args = SimpleNamespace(
        img_size=128,
        num_domains=3,
        latent_dim=16,
        style_dim=64,
        num_workers=0,
        seed=8365,
    )

    # Ruta al checkpoint
    checkpoint_path = "iter/20500_nets_ema.ckpt"

    # Crear la interfaz de Gradio
    inputs = [
        gr.Image(label="Source Image (Car to change style)"),
        gr.Image(label="Reference Image (Style to transfer)"),
    ]
    outputs = gr.Image(label="Generated Image (Car with transferred style)")

    title = "AutoStyleGAN: Car Style Transfer"
    description = "Transfer the style of one car to another. Upload a source car image and a reference car image."

    iface = gr.Interface(
        fn=lambda source_image, reference_image: main(source_image, reference_image, checkpoint_path, args),
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
    )
    return iface

if __name__ == '__main__':
    iface = gradio_interface()
    iface.launch(share=True)
