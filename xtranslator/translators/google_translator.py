from googletrans import Translator as google_translator
import asyncio
from typing import Union, List
import pandas as pd
from tqdm import tqdm
from .translator import Translator

class GoogleTranslator(Translator):
    def __init__(self, model_name: str, detector: str = None) -> None:
        self.model_name = model_name
        
        self.translator = google_translator()
        self.detector = None
    
    async def _translate_google_bulk(
        self,
        translator,
        texts: List[str],
        dest: str,
    ):
        results = [None] * len(texts)
        to_translate = []
        indices = []

        for i, text in enumerate(texts):
            to_translate.append(text)
            indices.append(i)

        if to_translate:
            translated = await translator.translate(to_translate, dest=dest)
            translated_texts = [t.text for t in translated]

            for idx, text, translated_text in zip(indices, to_translate, translated_texts):
                results[idx] = translated_text

        return results
    
    def translate(self, text: Union[str, List[str]], dest: str) -> Union[str, List[str]]:
        is_list = isinstance(text, list)
        
        if not is_list:
        
            output = asyncio.run(self.translator.translate(t, dest=dest))
            translated_text = output.text
            return translated_text
        else:
            return asyncio.run(
                self._translate_google_bulk(
                    self.translator,
                    text,
                    dest,
                )
            )
