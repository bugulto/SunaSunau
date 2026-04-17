import io
import os
import modal

APP_NAME = "nepali-asr-punc"

ASR_MODEL_DIR = "/models/models/asr2"
PUNC_MODEL_DIR = "/models/models/punc"
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_DIR = "/llama-model/Llama-3.1-8B-Instruct"
ADAPTER_MOUNT = "/adapter"
HF_HOME_DIR = "/llama-model/hf-cache"

LABELS = ["O", "PERIOD", "COMMA", "QUESTION"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("nepali-models", create_if_missing=True)
adapter_volume = modal.Volume.from_name("nepali-adapter", create_if_missing=True)
llama_volume = modal.Volume.from_name("llama3-1-8b-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
        "numpy==1.26.4",
        "pydub==0.25.1",
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "huggingface_hub==0.26.5",
        "sentencepiece==0.2.0",
        "safetensors==0.4.5",
        "protobuf==5.28.3",
    )
)


def _set_hf_env():
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("hf_token")
    )
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    os.environ.update({
        "HF_HOME": HF_HOME_DIR,
        "TRANSFORMERS_CACHE": HF_HOME_DIR,
        "HUGGINGFACE_HUB_CACHE": HF_HOME_DIR,
    })
    return token


def _find_adapter_dir(root: str = ADAPTER_MOUNT) -> str:
    if os.path.exists(os.path.join(root, "adapter_config.json")):
        return root
    for entry in os.scandir(root):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, "adapter_config.json")):
            return entry.path
    raise FileNotFoundError(f"No adapter_config.json found under {root}.")


def _tokenizer_source(adapter_dir: str) -> str:
    markers = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
    if any(os.path.exists(os.path.join(adapter_dir, m)) for m in markers):
        return adapter_dir
    return BASE_MODEL_DIR


def load_audio_16k_mono(audio_bytes: bytes):
    import numpy as np
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return samples / float(1 << (8 * audio.sample_width - 1))


@app.cls(image=image, gpu=None, timeout=300, scaledown_window=60, volumes={"/models": volume})
class ASRPunctuator:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.device = torch.device("cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_DIR, local_files_only=True)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_DIR, local_files_only=True).to(self.device).eval()
        self.punc_tokenizer = AutoTokenizer.from_pretrained(PUNC_MODEL_DIR, local_files_only=True)
        self.punc_model = AutoModelForTokenClassification.from_pretrained(PUNC_MODEL_DIR, local_files_only=True).to(self.device).eval()

    def _restore_punctuation(self, text: str) -> str:
        words = text.strip().split()
        if not words:
            return ""

        encoding = self.punc_tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        word_ids = encoding.word_ids()
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with self.torch.no_grad():
            preds = self.torch.argmax(self.punc_model(**encoding).logits, dim=-1)[0].cpu().numpy()

        suffix_map = {"PERIOD": "।", "COMMA": ",", "QUESTION": "?"}
        output_words, prev_word_id = [], None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                continue
            label = ID2LABEL.get(preds[token_idx], "O")
            output_words.append(words[word_id] + suffix_map.get(label, ""))
            prev_word_id = word_id

        return " ".join(output_words)

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        y = load_audio_16k_mono(audio_bytes)
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True)

        with self.torch.no_grad():
            logits = self.asr_model(inputs.input_values.to(self.device)).logits

        asr_text = self.processor.batch_decode(self.torch.argmax(logits, dim=-1))[0].strip()
        return {"asr_text": asr_text, "text": self._restore_punctuation(asr_text)}


@app.cls(
    image=image,
    gpu="T4",
    timeout=900,
    scaledown_window=100,
    volumes={ADAPTER_MOUNT: adapter_volume, "/llama-model": llama_volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
class NepaliChatBot:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        _set_hf_env()
        adapter_volume.reload()
        llama_volume.reload()

        adapter_dir = _find_adapter_dir(ADAPTER_MOUNT)

        self.tokenizer = AutoTokenizer.from_pretrained(
            _tokenizer_source(adapter_dir), use_fast=True, local_files_only=True, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
        )

        self.model = PeftModel.from_pretrained(
            base_model, adapter_dir, local_files_only=True, is_trainable=False
        ).eval()

        eos_ids = [self.tokenizer.eos_token_id]
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id != self.tokenizer.unk_token_id:
                eos_ids.append(eot_id)
        except Exception:
            pass
        seen: set[int] = set()
        self._eos_ids = [x for x in eos_ids if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    def _build_prompt(self, message: str, mode: str) -> tuple[str, dict]:
        configs = {
            "factual": (
                "तपाईं नेपालीमा छोटो, स्पष्ट र सिधा उत्तर दिने सहयोगी सहायक हुनुहुन्छ। "
                "प्रश्नको तथ्यपरक उत्तर दिनुहोस्। सकेसम्म पूर्ण वाक्यमा उत्तर दिनुहोस्। "
                "आवश्यक नभएसम्म लामो व्याख्या नगर्नुहोस्। "
                "उत्तर थाहा छैन भने अनुमान नगरी 'मलाई थाहा छैन।' भन्नुहोस्।",
                {"max_new_tokens": 80, "temperature": 0.3, "top_p": 0.85, "do_sample": True, "repetition_penalty": 1.15},
            ),
            "emotional": (
                "तपाईं नेपालीमा छोटो, नम्र र सहयोगी तरिकाले जवाफ दिने सहायक हुनुहुन्छ। "
                "मानवीय र सहानुभूतिपूर्ण शैलीमा उत्तर दिनुहोस्। उत्तर १ देखि २ वाक्यमा सीमित राख्नुहोस्। "
                "आवश्यक भएमा मात्र एउटा छोटो follow-up प्रश्न सोध्नुहोस्।",
                {"max_new_tokens": 70, "temperature": 0.35, "top_p": 0.85, "do_sample": True, "repetition_penalty": 1.15},
            ),
        }
        if mode not in configs:
            raise ValueError(f"mode must be 'factual' or 'emotional', got {mode!r}")

        system_prompt, gen_kwargs = configs[mode]
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": message.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, gen_kwargs

    def _decode(self, outputs, prompt_len: int) -> str:
        answer = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()

        for stop in ["<|eot_id|>", "<|end_of_text|>", "प्रयोगकर्ता:", "User:", "user:",
                     "assistant:", "Assistant:", "सहायक:", "System:", "system:", "###",
                     "\n\nप्रयोगकर्ता", "\n\nUser"]:
            if stop in answer:
                answer = answer.split(stop)[0].strip()

        answer = answer.split("\n\n")[0].strip()
        while "  " in answer:
            answer = answer.replace("  ", " ")
        return answer

    @modal.method()
    def chat(self, message: str, mode: str = "factual") -> dict:
        import torch

        prompt, gen_kwargs = self._build_prompt(message, mode)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, **gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self._eos_ids,
            )

        return {"text": self._decode(outputs, inputs["input_ids"].shape[-1]), "mode": mode}


@app.function(image=image, timeout=300)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel

    web = FastAPI()

    class ChatRequest(BaseModel):
        message: str
        mode: str = "factual"

    @web.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio payload")
        return await ASRPunctuator().transcribe.remote.aio(audio_bytes)

    @web.post("/chat")
    async def chat(payload: ChatRequest):
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        if payload.mode not in {"factual", "emotional"}:
            raise HTTPException(status_code=400, detail="mode must be factual or emotional")
        return await NepaliChatBot().chat.remote.aio(payload.message, payload.mode)

    return web

