import os
import torch

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import fix_state_dict_namespace
from gpt2_training.generation import generate_sequence


class GeneratorLM:
    def __init__(
        self,
        model_path,
        model_checkpoint,
        device="cpu",
        temperature=1,
        length=20,
        top_k=0,
        sampling=False,
        eos_id=50256,
        fp16=False,
    ):
        self.eos_id = eos_id
        self.eos_token = "<|endoftext|>"
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.sampling = sampling
        self.length = length
        self.fp16 = fp16

        self.init(model_path, model_checkpoint)

    def init(self, model_path, model_checkpoint):
        self.config = GPT2Config.from_json_file(os.path.join(model_path, "config.json"))
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel(self.config)

        model_state_dict = fix_state_dict_namespace(torch.load(model_checkpoint))

        start_model = self.model
        if hasattr(self.model, "transformer") and all(not s.startswith('transformer.') for s in model_state_dict.keys()):
            print('loading transfomer only')
            start_model = self.model.transformer
        start_model.load_state_dict(model_state_dict)

        if self.fp16:
            self.model.half()

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, text):
        # token ids
        text = text.lower()
        tokens = self.tokenizer.encode(text) + [self.eos_id]
        tokens = torch.tensor(tokens, device=self.device, dtype=torch.long)
        tokens = tokens.unsqueeze(0)

        # position ids
        position_ids = torch.arange(
            0, tokens.size(-1), device=self.device, dtype=torch.long
        )

        # token_type ids
        token_type_ids = torch.zeros_like(tokens, device=self.device, dtype=torch.long)
        return tokens, position_ids, token_type_ids

    def _postprocess(self, batch_output):
        texts = [self.tokenizer.decode(t) for t in batch_output.tolist()]
        texts = [t.split(self.eos_token)[0] for t in texts]
        return texts

    def generate(self, prompt):
        token_ids, position_ids, token_type_ids = self._preprocess(prompt)

        out = generate_sequence(
            self.model,
            token_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            length=self.length,
            start_token=None,
            temperature=self.temperature,
            top_k=self.top_k,
            sample=self.sampling,
        )

        return self._postprocess(out)
