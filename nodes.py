import os
import hashlib
from datetime import datetime
import json
import piexif
import piexif.helper
from PIL import Image, ExifTags, ImageOps, ImageSequence, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
import torch
from rembg import remove, new_session
import base64
from io import BytesIO
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from torchvision.transforms import functional as TF
import io
import subprocess
import time

matplotlib.use("Agg")


def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading the entire file into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp


def make_pathname(filename, seed, modelname, counter, time_format):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", modelname)
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    return filename


def make_filename(filename, seed, modelname, counter, time_format):
    filename = make_pathname(filename, seed, modelname, counter, time_format)

    return get_timestamp(time_format) if filename == "" else filename


class LoadImageWithMetaData:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "ImageSaverTools/utils"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width","height")
    FUNCTION = "load_image_with_metadata"
    def load_image_with_metadata(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        width, height = img.size
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, width, height)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class LoadImageWithCoordinates:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "ImageSaverTools/utils"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "load_image_with_coordinates"

    def load_image_with_coordinates(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        # Create a matplotlib figure and axes
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(img, aspect='equal')

        # Set no labels on axes
        ax.axis('on')  # 'on' to turn on axis lines and labels, 'off' to turn off
        ax.set_xticks(np.arange(0, width, 50))
        ax.set_yticks(np.arange(0, height, 50))
        ax.set_xticklabels([str(int(x)) for x in np.arange(0, width, 50)])
        ax.set_yticklabels([str(int(y)) for y in np.arange(0, height, 50)])
        ax.tick_params(axis='x', colors='red', direction='out', length=6, width=2)
        ax.tick_params(axis='y', colors='red', direction='out', length=6, width=2)
        ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5)

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Load image from buffer
        img_with_coords = Image.open(buf).convert('RGB')
        image = np.array(img_with_coords).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(image)[None,]

        # Create a dummy mask (since the original code expects a mask)
        mask = torch.zeros(1, img_tensor.shape[1], img_tensor.shape[2], dtype=torch.float32)

        return (img_tensor, mask, width, height)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        # input_dir = folder_paths.get_input_directory()
        # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image_base64": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "",
                    },
                ),
            }
        }

    CATEGORY = "ImageSaverTools/utils"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_base64"

    def load_image_from_base64(self, image_base64):
        # Decode the base64 string
        imgdata = base64.b64decode(image_base64)

        # Open the image from memory
        i = Image.open(BytesIO(imgdata))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image, mask)


class SeedGenerator:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_seed"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    def get_seed(self, seed):
        return (seed,)


class StringLiteral:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_string"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"default": "", "multiline": True})}}

    def get_string(self, string):
        return (string,)


class SizeLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8})}}

    def get_int(self, int):
        return (int,)


class WidthLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8})}}

    def get_int(self, int):
        return (int,)


class HeightLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8})}}

    def get_int(self, int):
        return (int,)


class StepsLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 20, "min": 1, "max": 100})}}

    def get_int(self, int):
        return (int,)


class IntLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 0, "min": 0, "max": 1000000})}}

    def get_int(self, int):
        return (int,)


class CfgLiteral:
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})}}

    def get_float(self, float):
        return (float,)


class CheckpointSelector:
    CATEGORY = 'ImageSaverTools/utils'
    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"),)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        cls.RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"),)
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),}}

    def get_names(self, ckpt_name):
        return (ckpt_name,)


class LoraSelector:
    CATEGORY = 'ImageSaverTools/utils'
    RETURN_TYPES = (folder_paths.get_filename_list("loras"),)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "get_names"

    # @classmethod
    # def RETURN_TYPES(self):
    #     return (folder_paths.get_filename_list("loras"),)

    @classmethod
    def INPUT_TYPES(cls):
        cls.RETURN_TYPES = (folder_paths.get_filename_list("loras"),)
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"), ),}}

    def get_names(self, lora_name):
        return (lora_name,)


class SamplerSelector:
    CATEGORY = 'ImageSaverTools/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (comfy.samplers.KSampler.SAMPLERS,)}}

    def get_names(self, sampler_name):
        return (sampler_name,)


class SchedulerSelector:
    CATEGORY = 'ImageSaverTools/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (comfy.samplers.KSampler.SCHEDULERS,)}}

    def get_names(self, scheduler):
        return (scheduler,)


class ImageSaveWithMetadata:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "modelname": (folder_paths.get_filename_list("checkpoints"),),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "positive": ("STRING", {"default": 'unknown', "multiline": True}),
                "negative": ("STRING", {"default": 'unknown', "multiline": True}),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "counter": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaverTools"

    def save_files(self, images, seed_value, steps, cfg, sampler_name, scheduler, positive, negative, modelname, lora_name, quality_jpeg_or_webp,
                   lossless_webp, width, height, counter, filename, path, extension, time_format, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed_value, modelname, counter, time_format)
        path = make_pathname(path, seed_value, modelname, counter, time_format)
        # ckpt_path = folder_paths.get_full_path("checkpoints", modelname)
        basemodelname = parse_name(modelname)
        lora_name = parse_name(lora_name)
        # modelhash = calculate_sha256(ckpt_path)[:10]
        modelhash = '0'
        comment = f"{handle_whitespace(positive)}\nNegative prompt: {handle_whitespace(negative)}\nSteps: {steps}, Sampler: {sampler_name}{f'_{scheduler}' if scheduler != 'normal' else ''}, CFG Scale: {cfg}, Seed: {seed_value}, Size: {width}x{height}, Model hash: {modelhash}, Model: {basemodelname}, Lora: {lora_name}, Version: ComfyUI"
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        filenames = self.save_images(images, output_path, filename, comment, extension, quality_jpeg_or_webp, lossless_webp, prompt, extra_pnginfo)

        subfolder = os.path.normpath(path)
        return {"ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)}}

    def save_images(self, images, output_path, filename_prefix, comment, extension, quality_jpeg_or_webp, lossless_webp, prompt=None, extra_pnginfo=None) -> list[str]:
        img_count = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            if extension == 'png':
                metadata = PngInfo()
                metadata.add_text("parameters", comment)

                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename = f"{filename_prefix}.png"
                img.save(os.path.join(output_path, filename), pnginfo=metadata)
            else:
                filename = f"{filename_prefix}.{extension}"
                file = os.path.join(output_path, filename)
                img.save(file, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(comment, encoding="unicode")
                    },
                })
                piexif.insert(exif_bytes, file)

            paths.append(filename)
            img_count += 1
        return paths


class ImageRemBG:
    session = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", )
    RETURN_NAMES = ("org_img", "image", "mask", )
    FUNCTION = "remove"

    CATEGORY = "ImageSaverTools/utils"

    def remove(self, image):
        session = self.get_session()
        image_np = image.numpy()
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8)[0])
        image_cut = remove(pil_image, session=session, bgcolor=(210,210,210,0))
        image_cut_out = np.array(image_cut).astype(np.float32) / 255.0
        image_cut_out = torch.from_numpy(image_cut_out)[None,]
        output = np.array(image_cut.convert('RGB')).astype(np.float32) / 255.0
        s = torch.from_numpy(output)[None,]
        mask = np.array(image_cut.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
        return (image_cut_out, s, mask, )

    def get_session(self):
        if not self.session:
            self.session = new_session()
        return self.session


class GaussianLatentImage:
    def __init__(self, device="cpu"):
        # self.device = device
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": 2048, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": 2048, "step": 1},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "ImageSaverTools/utils"

    def generate(self, width, height, batch_size=1, seed=0):
        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # Define the mean and standard deviation
        mean = 0
        var = 10
        sigma = var**0.5

        # Generate Gaussian noise
        gaussian = torch.randn((batch_size, 4, height // 8, width // 8)) * sigma + mean

        # Move the tensor to the specified device
        latent = gaussian.float().to(self.device)

        return ({"samples": latent},)


class GetImageResizeSize:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                "pixel": (
                    "INT",
                    {"default": 512, "min": 16, "max": 2048, "step": 1},
                )}}
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width_scale", "height_scale", "smaller_side", "larger_side", "width", 'height')
    FUNCTION = "get_size"

    CATEGORY = "ImageSaverTools/utils"

    def get_size(self, image, pixel):
        image_np = image.numpy()
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8)[0])
        width, height = pil_image.size
        width_scale, height_scale = self.calculate_closest_dimensions((width, height), pixel)
        width_1024, height_1024 = self.calculate_closest_dimensions((width, height), 1024)
        if height_1024 > width_1024:
            scale = width_1024 / width
            height_1024 = int(height * scale)
        else:
            scale = height_1024 / height
            width_1024 = int(width * scale)
        print(width_scale, height_scale, width_1024, height_1024)
        return (width_scale, height_scale, min(width_scale, height_scale), max(width_scale, height_scale), width_1024, height_1024, )

    def calculate_closest_dimensions(self, target_ratio, fixed_pixel_value):
        target_width, target_height = target_ratio
        total_share = target_width * target_height

        # 计算最接近的宽和高的像素值
        closest_width = round((fixed_pixel_value**2 / total_share) ** 0.5 * target_width)
        closest_height = round((fixed_pixel_value**2 / total_share) ** 0.5 * target_height)

        # 确保宽和高均为 8 的倍数
        closest_width = int(math.ceil(closest_width / 64) * 64)
        closest_height = int(math.ceil(closest_height / 64) * 64)

        return closest_width, closest_height


class SamLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("sams"), ),
        "model_type": ("STRING", {"default": "vit_h"})}}
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "load_checkpoint"

    CATEGORY = "ImageSaverTools/utils"

    def load_checkpoint(self, ckpt_name, model_type):
        from segment_anything import sam_model_registry, SamPredictor
        device = "cuda"
        ckpt_path = folder_paths.get_full_path("sams", ckpt_name)
        print(ckpt_path, 'loading sam')
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        sam.to(device=device)
        print('sam loaded')
        predictor = SamPredictor(sam)
        return (predictor, )


class SamDetect:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ('MODEL',),
        "image": ('IMAGE', ),
        "point_width": (
                    "INT",
                    {"default": 512, "min": 16, "max": 2048, "step": 1},
                ),
        "point_height": (
                    "INT",
                    {"default": 512, "min": 16, "max": 2048, "step": 1},
                )}}
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ('image', "mask", )
    FUNCTION = "sam_detect"

    CATEGORY = "ImageSaverTools/utils"

    def sam_detect(self, model, image, point_width, point_height):
        image_np = image.numpy()
        height, width = image.shape[1], image.shape[2]
        image = (image_np * 255).astype(np.uint8)[0]
        model.set_image(image)
        input_point = np.array([[point_width, point_height]])
        input_label = np.array([1])
        masks, scores, logits = model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        new_masks = []
        for mask, score in zip(masks, scores):
            mask = self.show_mask(mask)
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            ax.imshow(image)
            ax.imshow(mask)
            ax.set_title(str(score))
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            buf = Image.open(buf).convert('RGB')
            buf = np.array(buf).astype(np.float32) / 255.0
            new_masks.append(buf)
        new_masks = np.stack(new_masks, axis=0)
        new_masks = torch.from_numpy(new_masks)
        return_mask = torch.from_numpy(masks[np.argmax(scores)])
        return (new_masks, return_mask, )

    def show_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image


class BlipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("sams"), ),}}
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "load_checkpoint"

    CATEGORY = "ImageSaverTools/utils"

    def load_checkpoint(self, ckpt_name):
        from .blip.blip import BLIPCaptioner
        model = BLIPCaptioner(device='cuda')
        return (model, )


class BlipCaption:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ('MODEL',),
                             "image": ('IMAGE',),
                             'mask': ('MASK', )}}
    RETURN_TYPES = ("STRING", )
    FUNCTION = "main"

    CATEGORY = "ImageSaverTools/utils"

    def main(self, model, image, mask):
        image_np = image.numpy()
        image = (image_np * 255).astype(np.uint8)[0]
        mask = mask.numpy()
        print(mask.shape)
        caption = model.inference_with_reduced_tokens(image, mask)
        return (caption, )


class LlavaCaption:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ('IMAGE',),
                             'mask': ('MASK', )}}
    RETURN_TYPES = ("STRING", )
    FUNCTION = "main"

    CATEGORY = "ImageSaverTools/utils"

    def main(self, image, mask):
        image_np = image.numpy()
        image = (image_np * 255).astype(np.uint8)[0]
        mask = mask.numpy().astype(np.uint8)

        crop_save_path = self.generate_crop_image(image, mask)
        command = f"""CUDA_VISIBLE_DEVICES=1 /home/zxa/ps/llama_cpp/llava-cli -m /home/zxa/ps/pretrain_models/bunny/ggml-model-Q4_K_M.gguf --mmproj /home/zxa/ps/pretrain_models/bunny/mmproj-model-f16.gguf --image {crop_save_path} -c 4096 -p "Describe this image" --temp 0.0 -ngl 40 -n 2048"""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        caption = result.stdout.strip()
        return (caption, )

    def generate_crop_image(self, image, seg_mask):
        # image = np.array(image) * seg_mask[:, :, np.newaxis] + (1 - seg_mask[:, :, np.newaxis]) * 255
        seg_mask = np.array(seg_mask) > 0
        size = max(seg_mask.shape[0], seg_mask.shape[1])
        top, bottom = self.boundary(seg_mask)
        left, right = self.boundary(seg_mask.T)
        box = [left / size, top / size, right / size, bottom / size]
        size = max(image.shape[0], image.shape[1])
        x1, y1, x2, y2 = box
        image = Image.fromarray(image)
        image_crop = np.array(image.crop((x1 * size, y1 * size, x2 * size, y2 * size)))
        crop_save_path = f'/home/zxa/ps/http_deploy/generate_img/blip/result/crop_{time.time()}.png'
        Image.fromarray(image_crop).save(crop_save_path)

        print(f'crop save path {crop_save_path}')
        return crop_save_path

    def boundary(self, inputs):
        col = inputs.shape[1]
        inputs = inputs.reshape(-1)
        lens = len(inputs)
        start = np.argmax(inputs)
        end = lens - 1 - np.argmax(np.flip(inputs))
        top = start // col
        bottom = end // col
        return top, bottom


class CenterTransparentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ('IMAGE',)}}
    RETURN_TYPES = ("IMAGE", 'MASK')
    FUNCTION = "main"

    CATEGORY = "ImageSaverTools/utils"

    def main(self, image):
        image_np = image.numpy()
        image = (image_np * 255).astype(np.uint8)[0]
        image = Image.fromarray(image)
        
        # 获取图片尺寸
        width, height = image.size
        
        # 计算缩放比例
        scale = min(1024 / width, 1024 / height)
        
        # 调整图片尺寸
        new_size = (int(width * scale), int(height * scale))
        resized_image = image.resize(new_size, Image.LANCZOS)
        
        # 创建带透明度的白色背景
        background = Image.new('RGBA', (1024, 1024), (128, 128, 128, 255))
        
        # 计算图片放置位置
        offset = ((1024 - new_size[0]) // 2, (1024 - new_size[1]) // 2)
        
        # 创建mask图像
        mask = Image.new('L', (1024, 1024), 0)
        
        # 获取原始图片的alpha通道，并按比例调整大小
        alpha = resized_image.getchannel('A')
        mask.paste(alpha, offset)
        
        # 将图片放置在背景中
        background.paste(resized_image, offset, resized_image)
        
        background = background.convert('RGB')
        output_image = np.array(background).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image)[None,]
        
        # 创建掩码图像
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None, ...]

        return (output_image, mask)



NODE_CLASS_MAPPINGS = {
    "Checkpoint Selector": CheckpointSelector,
    "Save Image w/Metadata": ImageSaveWithMetadata,
    "Sampler Selector": SamplerSelector,
    "Scheduler Selector": SchedulerSelector,
    "Seed Generator": SeedGenerator,
    "String Literal": StringLiteral,
    "Width/Height Literal": SizeLiteral,
    'Width Literal': WidthLiteral,
    'Height Literal': HeightLiteral,
    'Steps Literal': StepsLiteral,
    "Cfg Literal": CfgLiteral,
    "Int Literal": IntLiteral,
    'Lora Selector': LoraSelector,
    "Load Image with Metadata": LoadImageWithMetaData,
    "Load Image with Cordinates": LoadImageWithCoordinates,
    "Remove Background": ImageRemBG,
    "LoadImageBase64": LoadImageBase64,
    'GaussianLatentImage': GaussianLatentImage,
    'Get Image Scaled Size': GetImageResizeSize,
    "Load Sam Checkpoint": SamLoader,
    "Sam Detect": SamDetect,
    "Load Blip": BlipLoader,
    "Blip Caption": BlipCaption,
    "Llava Caption": LlavaCaption,
    "Center Transparent Image": CenterTransparentImage
}
