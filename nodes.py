import os
import hashlib
from datetime import datetime
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION


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


def get_timestamp(time_format="%Y-%m-%d-%H%M%S"):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp


def make_filename(filename, seed, modelname, counter, time_format):
    timestamp = get_timestamp(time_format)

    # parse input string
    filename = filename.replace("%time", timestamp)
    filename = filename.replace("%model", modelname)
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))

    if filename == "":
        filename = timestamp
    return filename


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
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),}}

    def get_names(self, ckpt_name):
        return (ckpt_name,)


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
                        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                        "modelname": (folder_paths.get_filename_list("checkpoints"),),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    },
                    "optional": {
                        "positive": ("STRING", {"default": 'unknown', "multiline": True}),
                        "negative": ("STRING", {"default": 'unknown', "multiline": True}),
                        "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                        "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
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

    def save_files(self, images, seed_value, steps, cfg, sampler_name, scheduler, positive, negative, modelname,
                   width, height, counter, filename, path, time_format, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed_value, modelname, counter, time_format)
        ckpt_path = folder_paths.get_full_path("checkpoints", modelname)
        basemodelname = parse_name(modelname)
        modelhash = calculate_sha256(ckpt_path)[:10]
        comment = f"{handle_whitespace(positive)}\nNegative Prompt: {handle_whitespace(negative)}\nSteps: {steps}, Sampler: {sampler_name}, CFG Scale: {cfg}, Seed: {seed_value}, Size: {width}x{height}, Model hash: {modelhash}, Model: {basemodelname}, Version: ComfyUI, Scheduler: {scheduler}"
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        paths = self.save_images(images, output_path, filename, comment, prompt, extra_pnginfo)

        return {"ui": {"images": paths}}

    def save_images(self, images, output_path, filename_prefix="ComfyUI", comment="", prompt=None, extra_pnginfo=None):
        img_count = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
                metadata.add_text("parameters", comment)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            file = f"{filename_prefix}.png"
            img.save(os.path.join(output_path, file), pnginfo=metadata, optimize=True)

            paths.append(file)
            img_count += 1
        return paths


NODE_CLASS_MAPPINGS = {
    "Checkpoint Selector": CheckpointSelector,
    "Save Image w/Metadata": ImageSaveWithMetadata,
    "Sampler Selector": SamplerSelector,
    "Scheduler Selector": SchedulerSelector,
    "Seed Generator": SeedGenerator,
    "String Literal": StringLiteral,
    "Width/Height Literal": SizeLiteral,
    "Cfg Literal": CfgLiteral,
    "Int Literal": IntLiteral,
}