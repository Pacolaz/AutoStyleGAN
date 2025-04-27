import gradio as gr
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# DEFINICIÓN DE BLOQUES DE RED
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(dim_out, affine=True) if normalize else None
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(dim_out, affine=True) if normalize else None
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm1:
            out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)
        out = self.relu2(out)
        if self.downsample:
            out = self.avg_pool(out)
            residual = self.avg_pool(residual)
        out = out + residual
        return out

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, upsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(dim_out, style_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm2 = AdaIN(dim_out, style_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x, s):
        residual = x
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        out = self.conv1(x)
        out = self.norm1(out, s)
        out = self.relu1(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv2(out)
        out = self.norm2(out, s)
        out = self.relu2(out)
        out = out + residual
        return out

class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        gamma, beta = torch.chunk(h, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return (1 + gamma) * self.norm(x) + beta

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, style_dim, num_domains):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim + num_domains, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(512, style_dim)]

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.shared(h)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1) # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).unsqueeze(1).to(y.device)
        s = torch.gather(out, 1, idx.unsqueeze(2).expand(-1, -1, out.size(2))).squeeze(1)
        return s

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=3, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_in * (img_size // (2**repeat_num))**2, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1) # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).unsqueeze(1).to(y.device)
        s = torch.gather(out, 1, idx.unsqueeze(2).expand(-1, -1, out.size(2))).squeeze(1)
        return s

# DEFINICIÓN DEL GENERADOR
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

# FUNCIÓN PARA CARGAR EL MODELO
def load_pretrained_model(ckpt_path, img_size=256, style_dim=64, num_domains=3, device='cpu'):
    G = Generator(img_size, style_dim).to(device)
    M = MappingNetwork(16, style_dim, num_domains).to(device) # Suponiendo latent_dim=16
    S = StyleEncoder(img_size, style_dim, num_domains).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(checkpoint['generator'])
    M.load_state_dict(checkpoint['mapping_network'])
    S.load_state_dict(checkpoint['style_encoder'])
    G.eval()
    S.eval()
    return G, S

# FUNCIÓN PARA COMBINAR ESTILOS
def combine_styles(source_image, reference_image, generator, style_encoder, target_domain_idx, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Ajustar al tamaño de entrada de tu modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    source_img = transform(source_image).unsqueeze(0).to(device)
    reference_img = transform(reference_image).unsqueeze(0).to(device)
    target_domain = torch.tensor([target_domain_idx]).unsqueeze(0).to(device) # Crear un tensor para el dominio objetivo

    with torch.no_grad():
        style_ref = style_encoder(reference_img, target_domain) # Usar el mismo índice de dominio que la referencia
        generated_image = generator(source_img, style_ref)
        generated_image = (generated_image + 1) / 2.0 # Desnormalizar a [0, 1]
        generated_image = generated_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        generated_image = (generated_image * 255).astype(np.uint8)
        return Image.fromarray(generated_image)

# CONFIGURACIÓN DE GRADIO
def create_interface(generator, style_encoder, domain_names, device='cpu'):
    def predict(source_img, ref_img, target_domain):
        target_domain_idx = domain_names.index(target_domain)
        return combine_styles(source_img, ref_img, generator, style_encoder, target_domain_idx, device)

    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(label="Imagen Fuente"),
            gr.Image(label="Imagen de Referencia"),
            gr.Dropdown(choices=domain_names, label="Dominio de Referencia (para el estilo)"),
        ],
        outputs=gr.Image(label="Imagen Generada"),
        title="AutoStyleGAN - Transferencia de Estilo de Carros",
        description="Selecciona una imagen de carro fuente y una imagen de carro de referencia para transferir el estilo de la referencia a la fuente."
    )
    return iface

if __name__ == '__main__':
    #CARGAR EL MODELO ENTRENADO
    checkpoint_path = 'AutoStyleGAN/expr/checkpoints/9500_nets_ema.ckpt'
    img_size = 128
    style_dim = 64 
    num_domains = 3 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        generator, style_encoder = load_pretrained_model(checkpoint_path, img_size, style_dim, num_domains, device)
        print("Modelo cargado exitosamente.")

        #DEFINIR LOS NOMBRES DE LOS DOMINIOS
        # ¡Reemplaza con los nombres reales de tus dominios (carpetas)!
        domain_names = ["BMW", "Corvette", "Mazda"]

        #  CREAR E LANZAR LA INTERFAZ DE GRADIO 
        iface = create_interface(generator, style_encoder, domain_names, device)
        iface.launch(share=True)

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de checkpoint en '{checkpoint_path}'. Asegúrate de proporcionar la ruta correcta.")
    except Exception as e:
        print(f"Ocurrió un error al cargar el modelo: {e}")