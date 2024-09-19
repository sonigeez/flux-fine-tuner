import os
import time
from dataclasses import dataclass
from typing import List, Optional

import asyncio
from functools import lru_cache

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers.pipelines.flux import (
    FluxPipeline,
    FluxInpaintPipeline,
    FluxImg2ImgPipeline,
)

from weights import WeightsDownloadCache

import aiohttp
import aiofiles
import cv2

MODEL_URL_DEV = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
MODEL_URL_SCHNELL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/slim.tar"
FLUX_DEV_PATH = Path("FLUX.1-dev")
FLUX_SCHNELL_PATH = Path("FLUX.1-schnell")
FEATURE_EXTRACTOR = Path("/src/feature-extractor")

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

@dataclass
class LoadedLoRAs:
    main: Optional[str] = None
    extra: Optional[str] = None

class Predictor(BasePredictor):
    def setup(self) -> None:  # pyright: ignore
        """Initialize the weights cache and prepare for lazy loading of models."""
        start = time.time()
        # Don't pull weights
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.weights_cache = WeightsDownloadCache()

        # Initialize dictionaries for lazy loading
        self.pipes = {}
        self.img2img_pipes = {}
        self.inpaint_pipes = {}

        self.loaded_lora_urls = {
            "dev": LoadedLoRAs(),
            "schnell": LoadedLoRAs(),
        }
        print("Setup initiated")
        print("Setup took:", time.time() - start)

    async def async_load_model(self, session, url: str, path: Path, model_name: str) -> FluxPipeline:
        """Asynchronously download and load a single model."""
        if not path.exists():
            await self.download_base_weights(session, url, path)
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        return pipe

    async def download_and_load_pipes(self, model_key: str, url: str, path: Path, model_name: str):
        """Download and load pipelines for a specific model."""
        async with aiohttp.ClientSession() as session:
            pipe = await self.async_load_model(session, url, path, model_name)
            self.pipes[model_key] = pipe

            # Load img2img pipeline
            img2img_pipe = FluxImg2ImgPipeline(
                transformer=pipe.transformer,
                scheduler=pipe.scheduler,
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
            ).to("cuda")
            self.img2img_pipes[model_key] = img2img_pipe

            # Load inpaint pipeline
            inpaint_pipe = FluxInpaintPipeline(
                transformer=pipe.transformer,
                scheduler=pipe.scheduler,
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
            ).to("cuda")
            self.inpaint_pipes[model_key] = inpaint_pipe

            print(f"Loaded {model_key} pipelines")

    async def download_base_weights(self, session, url: str, dest: Path):
        """Asynchronously download model weights using aiohttp."""
        start = time.time()
        print("Downloading URL:", url)
        print("Downloading to:", dest)
        async with session.get(url) as resp:
            resp.raise_for_status()
            # Ensure the destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)
            f = await aiofiles.open(dest, mode='wb')
            async for chunk in resp.content.iter_chunked(1024):
                await f.write(chunk)
            await f.close()
        print("Download completed in:", time.time() - start)

    async def async_setup_pipes(self):
        """Asynchronously prepare any necessary resources for lazy loading."""
        # No models are loaded during setup; models are loaded on demand
        pass

    def setup_async(self):
        """Synchronous wrapper for asynchronous setup."""
        asyncio.run(self.async_setup_pipes())

    async def ensure_model_loaded(self, model: str):
        """Ensure that the requested model is loaded, loading it if necessary."""
        if model in self.pipes:
            print(f"Model '{model}' is already loaded.")
            return

        models_info = {
            "dev": (MODEL_URL_DEV, FLUX_DEV_PATH, "FLUX.1-dev"),
            "schnell": (MODEL_URL_SCHNELL, FLUX_SCHNELL_PATH, "FLUX.1-schnell"),
        }

        if model not in models_info:
            raise ValueError(f"Unknown model: {model}")

        url, path, model_name = models_info[model]
        print(f"Loading model '{model}'...")
        await self.download_and_load_pipes(model, url, path, model_name)

    @torch.inference_mode()
    def predict(  # pyright: ignore
        self,
        prompt: str = Input(
            description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image."
        ),
        image: Path = Input(
            description="Input image for img2img or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpainting mode. Black areas will be preserved, white areas will be inpainted. Must be provided along with 'image' for inpainting mode.",
            default=None,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image in text-to-image mode. The size will always be 1 megapixel, i.e. 1024x1024 if aspect ratio is 1:1. To use arbitrary width and height, set aspect ratio to 'custom'. Note: Ignored in img2img and inpainting modes.",
            choices=list(ASPECT_RATIOS.keys()) + ["custom"],  # pyright: ignore
            default="1:1",
        ),
        width: int = Input(
            description="Width of the generated image in text-to-image mode. Only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16). Note: Ignored in img2img and inpainting modes.",
            ge=256,
            le=1440,
            default=None,
        ),
        height: int = Input(
            description="Height of the generated image in text-to-image mode. Only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16). Note: Ignored in img2img and inpainting modes.",
            ge=256,
            le=1440,
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        lora_scale: float = Input(
            description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1.",
            default=1.0,
            le=2.0,
            ge=-1.0,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps. More steps can give more detailed images, but take longer.",
            ge=1,
            le=50,
            default=28,
        ),
        model: str = Input(
            description="Which model to run inferences with. The dev model needs around 28 steps but the schnell model only needs around 4 steps.",
            choices=["dev", "schnell"],
            default="dev",
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
            ge=0,
            le=10,
            default=3.5,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation.", default=None
        ),
        extra_lora: str = Input(
            description="Combine this fine-tune with another LoRA. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
            default=None,
        ),
        extra_lora_scale: float = Input(
            description="Determines how strongly the extra LoRA should be applied.",
            default=1.0,
            le=2.0,
            ge=-1.0,
        ),
        output_format: str = Input(
            description="Format of the output images.",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=90,
            ge=0,
            le=100,
        ),
        replicate_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Ensure the required model is loaded
        asyncio.run(self.ensure_model_loaded(model))

        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if aspect_ratio == "custom":
            if width is None or height is None:
                raise ValueError(
                    "width and height must be defined if aspect ratio is 'custom'"
                )
            width = make_multiple_of_16(width)
            height = make_multiple_of_16(height)
        else:
            width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        max_sequence_length = 512

        is_img2img_mode = image is not None and mask is None
        is_inpaint_mode = image is not None and mask is not None

        flux_kwargs = {}
        print(f"Prompt: {prompt}")

        if is_img2img_mode or is_inpaint_mode:
            # Use OpenCV for faster image processing
            input_image = cv2.imread(str(image))
            if input_image is None:
                raise ValueError(f"Cannot open image: {image}")
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            original_height, original_width = input_image.shape[:2]

            # Calculate dimensions that are multiples of 16
            target_width = make_multiple_of_16(original_width)
            target_height = make_multiple_of_16(original_height)
            target_size = (target_width, target_height)

            print(
                f"[!] Resizing input image from {original_width}x{original_height} to {target_width}x{target_height}"
            )

            # Determine if we should use highest quality settings
            use_highest_quality = output_quality == 100 or output_format == "png"

            # Resize the input image using OpenCV
            resampling_method = cv2.INTER_LANCZOS4 if use_highest_quality else cv2.INTER_CUBIC
            input_image = cv2.resize(input_image, target_size, interpolation=resampling_method)
            input_image = Image.fromarray(input_image)  # Convert back to PIL for compatibility
            flux_kwargs["image"] = input_image

            # Set width and height to match the resized input image
            flux_kwargs["width"], flux_kwargs["height"] = target_size

            if model not in self.pipes:
                raise ValueError(f"Model '{model}' is not loaded.")

            if is_img2img_mode:
                print("[!] img2img mode")
                pipe = self.img2img_pipes[model]
            else:  # is_inpaint_mode
                print("[!] inpaint mode")
                # Process mask image with OpenCV
                mask_image = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                if mask_image is None:
                    raise ValueError(f"Cannot open mask image: {mask}")
                mask_image = cv2.resize(mask_image, target_size, interpolation=cv2.INTER_NEAREST)
                mask_image = Image.fromarray(mask_image).convert("RGB")
                flux_kwargs["mask_image"] = mask_image
                pipe = self.inpaint_pipes[model]

            flux_kwargs["strength"] = prompt_strength
            print(
                f"[!] Using {model} model for {'img2img' if is_img2img_mode else 'inpainting'}"
            )
        else:  # is_txt2img_mode
            print("[!] txt2img mode")
            if model not in self.pipes:
                raise ValueError(f"Model '{model}' is not loaded.")
            pipe = self.pipes[model]
            flux_kwargs["width"] = width
            flux_kwargs["height"] = height

        if replicate_weights:
            flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}

        assert model in ["dev", "schnell"]
        if model == "dev":
            print("Using dev model")
            max_sequence_length = 512
        else:  # model == "schnell":
            print("Using schnell model")
            max_sequence_length = 256
            guidance_scale = 0

        if replicate_weights:
            start_time = time.time()
            if extra_lora:
                flux_kwargs["joint_attention_kwargs"] = {"scale": 1.0}
                print(f"Loading extra LoRA weights from: {extra_lora}")
                self.load_multiple_loras(replicate_weights, extra_lora, model)
                pipe.set_adapters(
                    ["main", "extra"], adapter_weights=[lora_scale, extra_lora_scale]
                )
            else:
                flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
                self.load_single_lora(replicate_weights, model)
                pipe.set_adapters(["main"], adapter_weights=[lora_scale])
            print(f"Loaded LoRAs in {time.time() - start_time:.2f}s")
        else:
            pipe.unload_lora_weights()
            self.loaded_lora_urls[model] = LoadedLoRAs(main=None, extra=None)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil",
        }

        # Batch inference
        output = pipe(**common_args, **flux_kwargs)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"./out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)  # type: ignore
            else:
                image.save(output_path)  # type: ignore
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "No output generated. Try running it again, or try a different prompt."
            )

        print(f"Generated {len(output_paths)} image(s).")
        return output_paths

    def load_single_lora(self, lora_url: str, model: str):
        # If no change, skip
        if lora_url == self.loaded_lora_urls[model].main:
            print("Weights already loaded")
            return

        pipe = self.pipes[model]
        pipe.unload_lora_weights()
        lora_path = self.weights_cache.ensure(lora_url)
        pipe.load_lora_weights(lora_path, adapter_name="main")
        self.loaded_lora_urls[model] = LoadedLoRAs(main=lora_url, extra=None)

    def load_multiple_loras(self, main_lora_url: str, extra_lora_url: str, model: str):
        pipe = self.pipes[model]
        loaded_lora_urls = self.loaded_lora_urls[model]

        # If no change, skip
        if (
            main_lora_url == loaded_lora_urls.main
            and extra_lora_url == loaded_lora_urls.extra
        ):
            print("Weights already loaded")
            return

        # Unload existing LoRA weights
        pipe.unload_lora_weights()

        # Load main LoRA
        main_lora_path = self.weights_cache.ensure(main_lora_url)
        pipe.load_lora_weights(main_lora_path, adapter_name="main")

        # Load extra LoRA
        extra_lora_path = self.weights_cache.ensure(extra_lora_url)
        pipe.load_lora_weights(extra_lora_path, adapter_name="extra")

        self.loaded_lora_urls[model] = LoadedLoRAs(
            main=main_lora_url, extra=extra_lora_url
        )

    @lru_cache(maxsize=None)  # Cached to avoid repeated calculations
    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

def make_multiple_of_16(n):
    # Rounds up to the next multiple of 16, or returns n if already a multiple of 16
    return ((n + 15) // 16) * 16

# --------------------------- Testing Harness Below ---------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path as SysPath  # To avoid confusion with cog.Path

    def parse_args():
        parser = argparse.ArgumentParser(description="Test Predictor Class")
        parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation.")
        parser.add_argument("--image", type=str, default=None, help="Path to input image for img2img or inpainting.")
        parser.add_argument("--mask", type=str, default=None, help="Path to mask image for inpainting.")
        parser.add_argument("--aspect_ratio", type=str, default="1:1", choices=list(ASPECT_RATIOS.keys()) + ["custom"], help="Aspect ratio for txt2img mode.")
        parser.add_argument("--width", type=int, default=None, help="Width for custom aspect ratio.")
        parser.add_argument("--height", type=int, default=None, help="Height for custom aspect ratio.")
        parser.add_argument("--num_outputs", type=int, default=1, choices=range(1,5), help="Number of images to generate.")
        parser.add_argument("--lora_scale", type=float, default=1.0, help="Scale for main LoRA.")
        parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps.")
        parser.add_argument("--model", type=str, default="dev", choices=["dev", "schnell"], help="Model to use.")
        parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for diffusion.")
        parser.add_argument("--prompt_strength", type=float, default=0.8, help="Prompt strength for img2img/inpaint.")
        parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
        parser.add_argument("--extra_lora", type=str, default=None, help="Extra LoRA URL.")
        parser.add_argument("--extra_lora_scale", type=float, default=1.0, help="Scale for extra LoRA.")
        parser.add_argument("--output_format", type=str, default="webp", choices=["webp", "jpg", "png"], help="Format of output images.")
        parser.add_argument("--output_quality", type=int, default=90, help="Quality of output images (0-100).")
        parser.add_argument("--replicate_weights", type=str, default=None, help="Replicate LoRA weights URL.")
        return parser.parse_args()

    async def run_prediction(predictor, args):
        """Asynchronously run the prediction."""
        # Convert SysPath to cog.Path if necessary
        image_path = Path(args.image) if args.image else None
        mask_path = Path(args.mask) if args.mask else None

        output_paths = predictor.predict(
            prompt=args.prompt,
            image=image_path,
            mask=mask_path,
            aspect_ratio=args.aspect_ratio,
            width=args.width,
            height=args.height,
            num_outputs=args.num_outputs,
            lora_scale=args.lora_scale,
            num_inference_steps=args.num_inference_steps,
            model=args.model,
            guidance_scale=args.guidance_scale,
            prompt_strength=args.prompt_strength,
            seed=args.seed,
            extra_lora=args.extra_lora,
            extra_lora_scale=args.extra_lora_scale,
            output_format=args.output_format,
            output_quality=args.output_quality,
            replicate_weights=args.replicate_weights,
        )

        print("Generated Images:")
        for path in output_paths:
            print(path)

    def main():
        args = parse_args()
        predictor = Predictor()
        print("Running setup...")
        predictor.setup_async()

        print("Running prediction...")
        asyncio.run(run_prediction(predictor, args))

    main()
