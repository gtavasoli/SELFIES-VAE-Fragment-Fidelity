"""Shared molecular VAE, reconstruction/KL losses, and token sampling.

Tensor dimensions use B for batch size, T for sequence length, V for vocabulary
size, and Z for latent dimension. Boolean padding masks use True for padding.
"""
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BaselineVAE", "top_p_filtering"]


def top_p_filtering(probs: torch.Tensor, top_p: float = 0.9, min_tokens: int = 1) -> torch.Tensor:
    """Filter and renormalize token probabilities using the recorded top-p rule.

    Parameters
    ----------
    probs : torch.Tensor
        Nonnegative probabilities of shape (B, V), normalized along the last axis.
    top_p : float, default=0.9
        Cumulative-probability cutoff in (0, 1].
    min_tokens : int, default=1
        Number of highest-probability tokens to retain regardless of the cutoff.

    Returns
    -------
    torch.Tensor
        Filtered probabilities of shape (B, V).

    Notes
    -----
    The historical rule retains sorted entries whose cumulative mass is at most
    ``top_p``, then forces the first ``min_tokens`` entries to remain.
    """
    # Sort by descending prob
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = cum <= top_p
    if min_tokens > 0:
        keep[:, :min_tokens] = True
    kept = sorted_probs * keep
    filtered = torch.zeros_like(probs)
    filtered.scatter_(1, sorted_idx, kept)
    denom = filtered.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return filtered / denom


class BaselineVAE(nn.Module):
    """Molecular VAE backbone for all three experiment configurations.

    Parameters
    ----------
    vocab_size : int
        Number of ordered vocabulary entries, including special tokens.
    pad_id, bos_id, eos_id : int
        Padding, beginning-of-sequence, and end-of-sequence token indices.
    emb : int, default=256
        Shared input/output embedding dimension.
    hid : int, default=512
        Encoder hidden dimension summed across both directions, and decoder
        hidden dimension. Must be even.
    z_dim : int, default=64
        Gaussian latent dimension.
    enc_layers, dec_layers : int, default=2 and 3
        Numbers of bidirectional encoder and unidirectional decoder LSTM layers.
    dropout : float, default=0.2
        Embedding/decoder-input dropout and inter-layer LSTM dropout.

    Notes
    -----
    The decoder concatenates a normalized latent vector at each token step and
    ties its vocabulary projection to the input embedding. ``forward`` returns
    unweighted reconstruction and KL losses; training loops set their weights
    and add fragment supervision for Contrastive-VAE and Rare-Aware-VAE.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        emb: int = 256,
        hid: int = 512,
        z_dim: int = 64,
        enc_layers: int = 2,
        dec_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hid % 2 == 0, "hid must be even (BiLSTM concatenates two halves)."

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.emb_dim = emb
        self.hid = hid
        self.z_dim = z_dim

        # ---- Embeddings (tie with output) ----
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=pad_id)

        # ---- Encoder: BiLSTM ----
        self.encoder = nn.LSTM(
            input_size=emb,
            hidden_size=hid // 2,         # per direction
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.enc_dropout = nn.Dropout(dropout)

        self.to_mu = nn.Linear(hid, z_dim)
        self.to_logvar = nn.Linear(hid, z_dim)

        # ---- Decoder: LSTM conditioned on z (concat at every step) ----
        self.dec_in_dim = emb + z_dim
        self.decoder = nn.LSTM(
            input_size=self.dec_in_dim,
            hidden_size=hid,
            num_layers=dec_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.dec_dropout = nn.Dropout(dropout)

        # ---- Output projection: tie with input embeddings ----
        self.out_proj = nn.Linear(hid, emb, bias=False)
        self.out_vocab = nn.Linear(emb, vocab_size, bias=False)
        # weight tying
        self.out_vocab.weight = self.embed.weight

        # Stabilize z scale
        self.z_norm = nn.LayerNorm(z_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize posterior and output-projection weights and posterior biases."""
        nn.init.xavier_uniform_(self.to_mu.weight)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.xavier_uniform_(self.to_logvar.weight)
        nn.init.zeros_(self.to_logvar.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # out_vocab tied to embed; no separate init

    # ---------- Encoder / Reparam ----------

    def encode(self, x: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode nonempty padded sequences into Gaussian posterior parameters.

        Parameters
        ----------
        x : torch.Tensor
            Integer token indices of shape (B, T).
        pad_mask : torch.Tensor
            Boolean mask of shape (B, T), with True at padded positions.

        Returns
        -------
        mu, logvar : torch.Tensor
            Posterior means and log variances, each of shape (B, Z).
        """
        emb = self.embed(x)  # [B, T, E]
        emb = self.enc_dropout(emb)

        lengths = (~pad_mask).sum(dim=1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.encoder(packed)
        # h_n: [layers*2, B, hid//2]; take last layer fwd/bwd
        h_fwd = h_n[-2]  # [B, hid//2]
        h_bwd = h_n[-1]  # [B, hid//2]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, hid]

        mu = self.to_mu(h_cat)
        logvar = self.to_logvar(h_cat)
        return mu, logvar

    # Alias explicitly for training loops (semantics identical; clearer intent)
    def encode_train(self, x: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``encode(x, pad_mask)`` for training-loop compatibility."""
        return self.encode(x, pad_mask)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``mu + exp(0.5 * logvar) * epsilon`` with standard-normal noise.

        Both inputs and the differentiable output have shape (B, Z).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- Decoder ----------

    def decode(self, x_in: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode teacher-forcing inputs conditioned on a latent vector.

        Parameters
        ----------
        x_in : torch.Tensor
            Integer input indices of shape (B, T), beginning with BOS.
        z : torch.Tensor
            Latent vectors of shape (B, Z).

        Returns
        -------
        torch.Tensor
            Unnormalized vocabulary logits of shape (B, T, V).
        """
        emb = self.embed(x_in)  # [B, T, E]
        z = self.z_norm(z)
        z_rep = z.unsqueeze(1).expand(-1, emb.size(1), -1)  # [B, T, Z]
        dec_in = torch.cat([emb, z_rep], dim=-1)            # [B, T, E+Z]
        dec_in = self.dec_dropout(dec_in)

        dec_out, _ = self.decoder(dec_in)                   # [B, T, H]
        h = self.out_proj(dec_out)                          # [B, T, E]
        logits = self.out_vocab(h)                          # [B, T, V]
        return logits

    # ---------- Forward ----------

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute stochastic reconstruction outputs and unweighted VAE losses.

        Parameters
        ----------
        x_in, x_out : torch.Tensor
            Shifted input/target token indices, each of shape (B, T).
        pad_mask : torch.Tensor
            Boolean target-padding mask of shape (B, T).

        Returns
        -------
        ce : torch.Tensor
            Scalar cross-entropy averaged over unmasked target positions.
        kl : torch.Tensor
            Scalar posterior-to-standard-normal KL, summed over Z and averaged
            over the batch.
        logits : torch.Tensor
            Decoder logits of shape (B, T, V).
        aux : dict
            Differentiable ``mu`` and ``logvar`` tensors of shape (B, Z).
        """
        mu, logvar = self.encode(x_in, pad_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(x_in, z)

        ce_tok = F.cross_entropy(
            logits.transpose(1, 2),  # [B, V, T]
            x_out,
            ignore_index=self.pad_id,
            reduction="none",
        )
        loss_mask = (~pad_mask).float()
        ce = (ce_tok * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        kl_el = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, Z]
        kl = kl_el.sum(dim=1).mean()

        aux = {"mu": mu, "logvar": logvar}
        return ce, kl, logits, aux

    # ---------- Sampling ----------

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        max_len: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Sample token sequences autoregressively without tracking gradients.

        Parameters
        ----------
        z : torch.Tensor
            Conditioning latent vectors of shape (B, Z).
        max_len : int
            Maximum emitted length, including BOS and EOS when sampled.
        temperature : float, default=1.0
            Divisor applied to logits before softmax.
        top_p : float, default=0.9
            Cutoff passed to ``top_p_filtering``.
        repetition_penalty : float, default=1.1
            Values above one subtract a count-based penalty for seen tokens.

        Returns
        -------
        torch.Tensor
            Integer indices of shape (B, T_generated), beginning with BOS.

        Notes
        -----
        Sampling stops when every row has emitted EOS at least once or the
        length limit is reached. A finished row can still emit later tokens;
        the tokenizer discards everything after its first EOS.
        """
        device = z.device
        B = z.size(0)
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        tokens = [cur]

        h_c = None
        z_n = self.z_norm(z).unsqueeze(1)

        for _ in range(max_len - 1):
            emb = self.embed(cur)                           # [B, 1, E]
            dec_in = torch.cat([emb, z_n], dim=-1)          # [B, 1, E+Z]
            out, h_c = self.decoder(dec_in, h_c)            # [B, 1, H]
            h = self.out_proj(out)                          # [B, 1, E]
            logits = self.out_vocab(h).squeeze(1)           # [B, V]

            logits = logits / max(1e-6, temperature)

            if len(tokens) > 1 and repetition_penalty > 1.0:
                seen = torch.cat(tokens, dim=1)             # [B, t]
                penal = torch.zeros_like(logits)
                penal = penal.scatter_add(1, seen, torch.ones_like(seen, dtype=logits.dtype))
                logits = logits - (repetition_penalty - 1.0) * penal

            probs = torch.softmax(logits, dim=-1)
            probs = top_p_filtering(probs, top_p=top_p, min_tokens=1)
            cur = torch.multinomial(probs, num_samples=1)   # [B, 1]
            tokens.append(cur)

            finished |= (cur.squeeze(1) == self.eos_id)
            if bool(finished.all()):
                break

        return torch.cat(tokens, dim=1)
