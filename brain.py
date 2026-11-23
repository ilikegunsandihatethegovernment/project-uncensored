import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectUncensoredLM(nn.Module):
    def __init__(self, vocab_size=256, block_size=256,
                 width=384, heads=6, layers=6):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)

        # Prevent each position from seeing future tokens.
        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt, max_new_bytes=300, temperature=0.8, top_k=40):
        self.eval()
        device = next(self.parameters()).device
        prompt_bytes = list(prompt.encode("utf-8")) or [10]
        tokens = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

        for _ in range(max_new_bytes):
            context = tokens[:, -self.block_size:]
            logits = self(context)[:, -1, :] / max(temperature, 0.01)

            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < threshold] = float("-inf")

            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        generated = bytes(tokens[0].tolist())
        return generated.decode("utf-8", errors="replace")
