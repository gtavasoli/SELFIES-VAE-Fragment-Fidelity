"""Ordered SELFIES vocabularies and shifted, padded sequence encoding."""
from typing import List, Tuple

class SelfiesTokenizer:
    """Map SELFIES tokens to stable indices and prepare teacher-forcing pairs.

    Parameters
    ----------
    vocab_path : str or path-like
        UTF-8 file with one token per nonempty line. The first four entries must
        be ``<pad>``, ``<bos>``, ``<eos>``, and ``<unk>`` in that order.

    Attributes
    ----------
    id_to_tok : list of str
        Vocabulary in checkpoint-compatible index order.
    tok_to_id : dict
        Reverse token-to-index mapping.

    Raises
    ------
    ValueError
        If special-token ordering is invalid or the vocabulary has duplicates.
    """
    def __init__(self, vocab_path: str):
        tokens = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    tokens.append(t)
        self._set_tokens(tokens)

    @classmethod
    def from_tokens(cls, tokens):
        """Build a tokenizer from an ordered iterable instead of a vocab file.

        The same ordering and uniqueness checks as the constructor apply.
        """
        tokenizer = cls.__new__(cls)
        tokenizer._set_tokens(list(tokens))
        return tokenizer

    def _set_tokens(self, tokens):
        """Validate the ordered vocabulary and initialize index mappings."""
        if tokens[:4] != ['<pad>', '<bos>', '<eos>', '<unk>']:
            raise ValueError('Vocabulary must start with <pad>,<bos>,<eos>,<unk>.')
        if len(tokens) != len(set(tokens)):
            raise ValueError('Vocabulary contains duplicate tokens.')
        self.id_to_tok = tokens
        self.tok_to_id = {t:i for i,t in enumerate(tokens)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0,1,2,3

    def encode(self, selfies_str: str, max_len: int) -> Tuple[List[int], List[int], List[bool]]:
        """Create fixed-length input/target sequences and a target-padding mask.

        Parameters
        ----------
        selfies_str : str
            Molecular or fragment SELFIES to split into tokens.
        max_len : int
            Length of each output list, including added special tokens.

        Returns
        -------
        ids_in, ids_out : list of int
            BOS-prefixed inputs and EOS-suffixed targets, truncated and padded
            independently to ``max_len``. Truncation can remove EOS.
        pad_mask : list of bool
            True where ``ids_out`` contains padding. Unknown tokens map to UNK.
        """
        import selfies as sf
        toks = list(sf.split_selfies(selfies_str))
        ids = [self.tok_to_id.get(t, self.unk_id) for t in toks]
        ids_in  = ([self.bos_id] + ids)[:max_len]
        ids_out = (ids + [self.eos_id])[:max_len]
        if len(ids_in) < max_len:  ids_in += [self.pad_id]*(max_len-len(ids_in))
        if len(ids_out) < max_len: ids_out += [self.pad_id]*(max_len-len(ids_out))
        pad_mask = [tid==self.pad_id for tid in ids_out]
        return ids_in, ids_out, pad_mask

    def decode_ids(self, ids: List[int]) -> str:
        """Join token indices into SELFIES, skipping PAD/BOS and stopping at EOS.

        UNK entries are retained as their literal token text. Indices must be
        valid entries in this tokenizer's ordered vocabulary.
        """
        out=[]
        for i in ids:
            if i==self.eos_id: break
            if i in (self.pad_id, self.bos_id): continue
            out.append(self.id_to_tok[i])
        return "".join(out)
