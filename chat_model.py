from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
import torch

from emu3.mllm.processing_emu3 import Emu3Processor

class ChatModel():
    def __init__(self):
        self.EMU_HUB = "/root/.cache/modelscope/hub/BAAI/Emu3-Chat"
        self.VQ_HUB = "/root/.cache/modelscope/hub/BAAI/Emu3-VisionTokenizer"
        self.DEVICE = "cuda:5"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.EMU_HUB,
            device_map=self.DEVICE,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.EMU_HUB, trust_remote_code=True, padding_side="left")
        self.image_processor = AutoImageProcessor.from_pretrained(self.VQ_HUB, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.VQ_HUB, device_map=self.DEVICE, trust_remote_code=True).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)
    def chat(self,text,image):
        inputs = self.processor(
            text=text,
            image=image,
            mode='U',
            return_tensors="pt",
            padding="longest",
        )
        GENERATION_CONFIG = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
        )
        outputs = self.model.generate(
            inputs.input_ids.to(self.DEVICE),
            GENERATION_CONFIG,
            attention_mask=inputs.attention_mask.to(self.DEVICE),
        )
        outputs = outputs[:, inputs.input_ids.shape[-1]:]
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
