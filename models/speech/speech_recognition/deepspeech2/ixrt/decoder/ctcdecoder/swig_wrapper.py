"""
CTC decoder backend using pyctcdecode + kenlm.

This module replaces the original paddlespeech_ctcdecoders SWIG extension with
a pure-Python equivalent backed by pyctcdecode, preserving the same public API
that CTCDecoder (ctc.py) relies on.
"""

import numpy as np

try:
    from pyctcdecode import build_ctcdecoder as _build_ctcdecoder
    _PYCTCDECODE_AVAILABLE = True
except ImportError:
    _PYCTCDECODE_AVAILABLE = False
    print("[swig_wrapper] Warning: pyctcdecode not found, falling back to greedy decoding.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_labels(vocab_list):
    """Map the project's VOCABLIST tokens to pyctcdecode-compatible labels.

    pyctcdecode conventions:
    - "" (empty string) at the blank position (index 0)
    - " " (space) for word-boundary tokens
    - everything else kept as-is
    """
    labels = []
    for tok in vocab_list:
        if tok in ("<blank>", "<pad>"):
            labels.append("")          # blank token
        elif tok == "<space>":
            labels.append(" ")         # word boundary
        elif tok.startswith("<") and tok.endswith(">"):
            labels.append("")          # other special tokens → treated as blank
        else:
            labels.append(tok)
    return labels


def _greedy_decode(probs_seq, vocabulary, blank_id):
    """Pure-Python CTC greedy decoder.

    Args:
        probs_seq: 2-D list or array [T, vocab_size]
        vocabulary: list of token strings
        blank_id: index of the blank token

    Returns:
        Decoded string.
    """
    arr = np.array(probs_seq, dtype=np.float32)
    best_path = np.argmax(arr, axis=-1)

    tokens = []
    prev = -1
    for idx in best_path:
        if idx != prev:
            if idx != blank_id:
                tokens.append(vocabulary[idx])
            prev = idx
        else:
            prev = idx

    text = ""
    for tok in tokens:
        if tok == "<space>":
            text += " "
        elif tok not in ("<blank>", "<unk>", "<eos>", "<pad>"):
            text += tok
    return text.strip()


# ---------------------------------------------------------------------------
# Scorer  (LM scorer stub — parameters are consumed by CTCBeamSearchDecoder)
# ---------------------------------------------------------------------------

class Scorer:
    """Language model scorer metadata holder.

    The original paddlespeech_ctcdecoders.Scorer was a C++ object that loaded
    a KenLM model.  Here we store the parameters so that
    CtcBeamSearchDecoderBatch can pick them up when building the pyctcdecode
    decoder instance.
    """

    def __init__(self, alpha, beta, model_path, vocabulary):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Methods called by CTCDecoder._init_ext_scorer (for logging only)
    # ------------------------------------------------------------------
    def is_character_based(self):
        return True

    def get_max_order(self):
        return 5

    def get_dict_size(self):
        return 0

    def reset_params(self, alpha, beta):
        self.alpha = float(alpha)
        self.beta = float(beta)


# ---------------------------------------------------------------------------
# CtcBeamSearchDecoderBatch  (stateful, one call of next() + decode() per sample)
# ---------------------------------------------------------------------------

class CtcBeamSearchDecoderBatch:
    """Batched CTC beam-search decoder backed by pyctcdecode.

    The original C++ class was stateful: next() fed frame probabilities and
    decode() returned the accumulated result.  Because inference.py calls
    next() exactly once per sample before decode(), we simply store the probs
    and run the full decode in decode().
    """

    def __init__(self, vocab_list, batch_size, beam_size, num_processes,
                 cutoff_prob, cutoff_top_n, ext_scorer, blank_id):
        self.batch_size = int(batch_size)
        self.beam_size = int(beam_size)
        self.blank_id = int(blank_id)
        self._vocab_list = list(vocab_list)

        self._pending_probs = [None] * self.batch_size

        labels = _make_labels(vocab_list)

        # Try to build a pyctcdecode decoder with the KenLM language model.
        self._ctc_decoder = None
        if _PYCTCDECODE_AVAILABLE:
            try:
                lm_path = getattr(ext_scorer, "model_path", None)
                alpha = getattr(ext_scorer, "alpha", 0.0)
                beta = getattr(ext_scorer, "beta", 0.0)

                if lm_path:
                    self._ctc_decoder = _build_ctcdecoder(
                        labels=labels,
                        kenlm_model_path=lm_path,
                        alpha=alpha,
                        beta=beta,
                    )
                    print(f"[swig_wrapper] KenLM decoder built: alpha={alpha}, beta={beta}")
                else:
                    self._ctc_decoder = _build_ctcdecoder(labels=labels)
                    print("[swig_wrapper] pyctcdecode decoder built (no LM).")
            except Exception as e:
                print(f"[swig_wrapper] Warning: pyctcdecode init failed ({e}), "
                      "using greedy decoding.")
                self._ctc_decoder = None

    # ------------------------------------------------------------------
    # Stateful interface
    # ------------------------------------------------------------------

    def next(self, probs_split, has_value):
        """Receive probability frames for this batch.

        Args:
            probs_split: list of B elements, each is a list[list[float]]
                         of shape [T, vocab_size].
            has_value:   list of B "true"/"false" strings (ignored here).
        """
        for i, probs in enumerate(probs_split):
            if i < self.batch_size:
                self._pending_probs[i] = probs

    def decode(self):
        """Run CTC decoding on stored probabilities.

        Returns:
            List[List[Tuple[float, str]]]: outer list = batch, inner list = beam
                candidates, each candidate is (score, text_string).
        """
        results = []
        for i in range(self.batch_size):
            probs = self._pending_probs[i]
            if probs is None:
                results.append([(0.0, "")])
                continue

            probs_arr = np.array(probs, dtype=np.float32)

            if self._ctc_decoder is not None:
                # pyctcdecode expects log-probabilities [T, vocab_size]
                log_probs = np.log(np.clip(probs_arr, 1e-10, 1.0))
                text = self._ctc_decoder.decode(
                    log_probs, beam_width=min(self.beam_size, 100)
                )
            else:
                text = _greedy_decode(probs, self._vocab_list, self.blank_id)

            results.append([(0.0, text)])

        return results

    def reset_state(self, batch_size=-1, beam_size=-1, num_processes=-1,
                    cutoff_prob=-1.0, cutoff_top_n=-1):
        """Reset internal buffer so the next sample starts fresh."""
        if batch_size > 0:
            self.batch_size = batch_size
        if beam_size > 0:
            self.beam_size = beam_size
        self._pending_probs = [None] * self.batch_size


# ---------------------------------------------------------------------------
# CTCBeamSearchDecoder  (inherits CtcBeamSearchDecoderBatch, as in the original)
# ---------------------------------------------------------------------------

class CTCBeamSearchDecoder(CtcBeamSearchDecoderBatch):
    """Drop-in replacement for paddlespeech_ctcdecoders.CtcBeamSearchDecoderBatch."""

    def __init__(self, vocab_list, batch_size, beam_size, num_processes,
                 cutoff_prob, cutoff_top_n, ext_scorer, blank_id):
        super().__init__(vocab_list, batch_size, beam_size, num_processes,
                         cutoff_prob, cutoff_top_n, ext_scorer, blank_id)


# ---------------------------------------------------------------------------
# Standalone decoding functions (used by _decode_batch_* deprecated paths)
# ---------------------------------------------------------------------------

def ctc_greedy_decoding(probs_seq, vocabulary, blank_id):
    """CTC greedy decoding for a single utterance.

    Args:
        probs_seq: 2-D list [T, vocab_size]
        vocabulary: list of token strings
        blank_id: index of the blank token

    Returns:
        Decoded string.
    """
    return _greedy_decode(probs_seq, vocabulary, blank_id)


def ctc_beam_search_decoding(probs_seq, vocabulary, beam_size,
                              cutoff_prob=1.0, cutoff_top_n=40,
                              ext_scoring_func=None, blank_id=0):
    """CTC beam search decoding for a single utterance.

    Returns a list of (score, bytes) tuples compatible with the original API.
    """
    text = _greedy_decode(probs_seq, vocabulary, blank_id)
    return [(0.0, text.encode("utf-8"))]


def ctc_beam_search_decoding_batch(probs_split, vocabulary, beam_size,
                                    num_processes, cutoff_prob=1.0,
                                    cutoff_top_n=40, ext_scoring_func=None,
                                    blank_id=0):
    """CTC beam search decoding for a batch of utterances.

    Returns a list of list of (score, str) tuples compatible with the
    original API.
    """
    results = []
    for probs_seq in probs_split:
        text = _greedy_decode(probs_seq, vocabulary, blank_id)
        results.append([(0.0, text)])
    return results