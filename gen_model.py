from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch

from emu3.mllm.processing_emu3 import Emu3Processor

class GenModel():
    def __init__(self):
        self.EMU_HUB = "/root/.cache/modelscope/hub/BAAI/Emu3-Gen"
        self.VQ_HUB = "/root/.cache/modelscope/hub/BAAI/Emu3-VisionTokenizer"
        self.DEVICE = "cuda:1"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.EMU_HUB,
            device_map=self.DEVICE,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.EMU_HUB, trust_remote_code=True, padding_side="left")
        self.image_processor = AutoImageProcessor.from_pretrained(self.VQ_HUB, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.VQ_HUB, device_map=self.DEVICE, trust_remote_code=True).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        self.POSITIVE_PROMPT = " masterpiece, film grained, best quality."
        self.NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

        self.classifier_free_guidance = 3.0
    def format_prompt(self,prompt):
        prompt = [prompt + self.POSITIVE_PROMPT]
        return prompt
    def gen_image(self,prompt,ratios):
        prompt = self.format_prompt(prompt)

        kwargs = dict(
            mode='G',
            ratio = ratios,
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = self.processor(text=prompt, **kwargs)
        neg_inputs = self.processor(text=[self.NEGATIVE_PROMPT] * len(prompt), **kwargs)

        GENERATION_CONFIG = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                self.classifier_free_guidance,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to(self.DEVICE),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn ,
                num_beams=1,
            ),
        ])

        outputs = self.model.generate(
            pos_inputs.input_ids.to(self.DEVICE),
            GENERATION_CONFIG,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(self.DEVICE),
        )

        mm_list = self.processor.decode(outputs[0])
        return mm_list, outputs[0]