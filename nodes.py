import os
import torch
import numpy as np
import comfy.model_management as mm
from PIL import Image
import folder_paths
from transformers import AutoModelForCausalLM, AutoProcessor

# Helper function to convert ComfyUI tensor image to PIL Image
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Helper function to convert PIL Image to ComfyUI tensor image (if needed, though not used in ApplyKimiVL output)
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class KimiVLModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Determine if flash-attn is available
        try:
            import flash_attn
            flash_attn_available = True
        except ImportError:
            flash_attn_available = False
            print("Flash Attention 2 not found. Install it for better performance: pip install flash-attn --no-build-isolation")

        # Define dtype choices, prioritize bfloat16 if available
        dtype_choices = ["auto", "fp16", "bf16", "fp32"]
        default_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

        return {
            "required": {
                "model_path": ("STRING", {"default": "moonshotai/Kimi-VL-A3B-Instruct"}),
                "dtype": (dtype_choices, {"default": default_dtype}),
                "use_flash_attention_2": ("BOOLEAN", {"default": flash_attn_available and default_dtype == "bf16"}), # Only default true if available and bf16
            }
        }

    RETURN_TYPES = ("MODEL", "PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "KimiVL"

    def load_model(self, model_path, dtype, use_flash_attention_2):
        device = mm.get_torch_device()
        print(f"KimiVL: Loading model '{model_path}' to device '{device}' with dtype '{dtype}'. Flash Attention 2: {use_flash_attention_2}")

        # Determine torch dtype
        torch_dtype = torch.float32
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                print("Warning: bf16 selected but not supported by hardware. Falling back to fp16.")
                torch_dtype = torch.float16
        elif dtype == "auto":
            # Let transformers decide, but prefer bfloat16 if supported
             torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


        attn_implementation = None
        if use_flash_attention_2:
            try:
                import flash_attn
                if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
                     attn_implementation="flash_attention_2"
                     print("KimiVL: Using Flash Attention 2.")
                else:
                    print("Warning: Flash Attention 2 requested but dtype is not bf16 or fp16. Disabling.")
            except ImportError:
                print("Warning: Flash Attention 2 requested but not installed. Install it via: pip install flash-attn --no-build-isolation")

        # --- Model Loading ---
        print(f"KimiVL: Loading model using dtype: {torch_dtype}, attn_implementation: {attn_implementation}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto", # Let accelerate handle device placement
            trust_remote_code=True,
            attn_implementation=attn_implementation
        ).eval() # Set to evaluation mode

        # --- Processor Loading ---
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"KimiVL: Model '{model_path}' loaded successfully.")
        return (model, processor)


class ApplyKimiVL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "processor": ("PROCESSOR",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "What is the dome building in the picture? Think step by step."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Add seed for reproducibility if needed later
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "apply_kimi_vl"
    CATEGORY = "KimiVL"

    def apply_kimi_vl(self, model, processor, image, prompt, max_new_tokens, seed):
        device = mm.get_torch_device() # Although model is on device via device_map, inputs need to be moved
        # print(f"ApplyKimiVL: Using device {device} from model: {model.device}") # model.device should be correct due to device_map

        # Set seed for reproducibility if generation options support it (e.g., do_sample=True)
        # torch.manual_seed(seed) # Kimi example uses greedy search (do_sample=False), so seed might not have effect

        # 1. Convert ComfyUI IMAGE tensor to PIL Image
        image_pil = tensor2pil(image)

        # 2. Prepare messages for the chat template
        #    Note: The original script passes the PIL image directly to the processor later,
        #          and uses apply_chat_template only for the text part. Let's follow that.
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]} # Image is handled separately by processor
        ]

        # 3. Apply chat template to the text part
        #    Important: Kimi's processor expects the image separately in the next step.
        #    We only template the text message here.
        try:
            text_prompt_templated = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) # Get templated string
             # print(f"ApplyKimiVL: Templated text prompt: {text_prompt_templated}") # Debugging
        except Exception as e:
             print(f"Error during apply_chat_template: {e}")
             # Fallback or re-raise depending on desired robustness
             # Fallback to raw prompt if template fails? Or just error out? Let's error out for now.
             raise e


        # 4. Process image and templated text together using the processor
        #    Pass the PIL image here.
        try:
            inputs = processor(
                text=text_prompt_templated,
                images=image_pil,
                return_tensors="pt",
                padding=True, # Ensure padding is handled if batching (though here it's single)
                truncation=True
            ).to(model.device) # Move inputs to the same device as the model
            # print(f"ApplyKimiVL: Inputs prepared for device: {model.device}") # Debugging
            # print(f"ApplyKimiVL: Input IDs shape: {inputs['input_ids'].shape}") # Debugging
        except Exception as e:
            print(f"Error during processor call: {e}")
            raise e


        # 5. Generate text
        # Ensure inputs are on the correct device (should be handled by .to(model.device) above)
        # print(f"ApplyKimiVL: Generating with max_new_tokens={max_new_tokens}") # Debugging
        with torch.inference_mode():
            try:
                # Original example uses greedy search (no sampling)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    # do_sample=False # Default for generate is greedy
                    # Other potential args: temperature, top_p, top_k if do_sample=True
                )
                # print(f"ApplyKimiVL: Generated IDs shape: {generated_ids.shape}") # Debugging
            except Exception as e:
                print(f"Error during model.generate: {e}")
                # Attempt to clear cache and raise
                mm.soft_empty_cache()
                raise e


        # 6. Trim input tokens from the generated sequence
        input_token_len = inputs.input_ids.shape[1]
        # print(f"ApplyKimiVL: Input token length: {input_token_len}") # Debugging
        generated_ids_trimmed = generated_ids[:, input_token_len:]
        # print(f"ApplyKimiVL: Trimmed generated IDs shape: {generated_ids_trimmed.shape}") # Debugging


        # 7. Decode the generated tokens
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print(f"ApplyKimiVL: Decoded response: {response}") # Debugging

        # Clean up memory (optional, but good practice for large models)
        # del inputs
        # del generated_ids
        # mm.soft_empty_cache() # Moved outside the main logic if needed

        return (response,) # Return as a tuple


# --- ComfyUI Registration ---

NODE_CLASS_MAPPINGS = {
    "KimiVLModelLoader": KimiVLModelLoader,
    "ApplyKimiVL": ApplyKimiVL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimiVLModelLoader": "Load Kimi-VL Model",
    "ApplyKimiVL": "Apply Kimi-VL (Image & Text)"
}

print("------------------------------------------")
print("ComfyUI Kimi-VL Nodes loaded successfully!")
print("------------------------------------------")
