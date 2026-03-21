import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import Union, List, Tuple
from tqdm import tqdm
from xtranslator.detectors import load_detector
from .translator import Translator
from .utils import NLLB_LANGS


class TranslateGemmaTranslator(Translator):
    def __init__(self, model_name: str, detector: str) -> None:
        self.model_name = model_name
        self._check_model()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = load_detector(detector)

    def _check_model(self):
        available_models = [
            'google/translategemma-4b-it',
            'google/translategemma-12b-it',
            'google/translategemma-27b-it',
        ]

        if self.model_name not in available_models:
            raise ValueError(f'{self.model_name} is not available. Please choose from {available_models}')

    def _get_language_code(self, text: str) -> Tuple[str, str]:
        language = self.detector.detect(text)
        return language

    def translate(self, text: Union[str, List[str]], dest: str) -> Union[str, List[str]]:
        is_list = isinstance(text, list)

        if not is_list:
            text = [text]

        translated = []

        for t in tqdm(text, desc='Translating', total=len(text)):
            lang = self._get_language_code(t)

            if  lang == dest:
                translated.append(t)
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": lang,
                            "target_lang_code": dest,
                            "text": t,
                        }
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            input_len = len(inputs['input_ids'][0])

            with torch.inference_mode():
                generation = self.model.generate(**inputs, do_sample=False, max_new_tokens=2048)

            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            translated.append(decoded)

        if is_list:
            return translated
        else:
            return translated[0]
