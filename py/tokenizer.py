"""
Unified tokenizer interface for character-level and BPE tokenization.

Provides CharTokenizer (existing character-level approach) and BPETokenizer
(using tiktoken for 16K vocabulary) with identical interfaces.
"""

import pickle
from abc import ABC, abstractmethod
from typing import List, Dict
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import WordPiece as WordPieceDecoder


class Tokenizer(ABC):
    """Abstract base class for tokenizers"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer state to file"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer state from file"""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size"""
        pass
    
    @property
    @abstractmethod
    def tokenizer_type(self) -> str:
        """Return tokenizer type identifier"""
        pass


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    Maps each unique character to an integer ID.
    """
    
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._vocab_size = 0
    
    def train(self, text: str) -> None:
        """Build vocabulary from text"""
        chars = sorted(list(set(text)))
        self._vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        print(f"CharTokenizer: Built vocabulary of {self._vocab_size} characters")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs, using 0 for unknown characters"""
        return [self.stoi.get(c, 0) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text, using '?' for unknown IDs"""
        return ''.join([self.itos.get(i, '?') for i in tokens])
    
    def save(self, path: str) -> None:
        """Save vocabulary mappings"""
        data = {
            'tokenizer_type': 'char',
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self._vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load vocabulary mappings"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        assert data['tokenizer_type'] == 'char', "Not a character tokenizer file"
        self.stoi = data['stoi']
        self.itos = data['itos']
        self._vocab_size = data['vocab_size']
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def tokenizer_type(self) -> str:
        return 'char'


class BPETokenizer(Tokenizer):
    """
    BPE tokenizer using HuggingFace tokenizers library.
    Trains a custom vocabulary of specified size on provided corpus.
    """
    
    def __init__(self, vocab_size: int = 16384):
        self.target_vocab_size = vocab_size
        self.tokenizer: HFTokenizer = None
        self._vocab_size = 0
    
    def train(self, text: str) -> None:
        """
        Train BPE vocabulary on text.
        Uses HuggingFace tokenizers library with ByteLevel pre-tokenizer
        which preserves all characters including spaces.
        """
        print(f"BPETokenizer: Training vocabulary of {self.target_vocab_size} tokens...")
        print(f"Training on {len(text):,} characters...")
        
        # Initialize BPE tokenizer
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        
        # Use ByteLevel pre-tokenizer (like GPT-2) which preserves all bytes
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        # Set up BPE trainer
        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=["[UNK]", "<|endoftext|>"],
            show_progress=True
        )
        
        # Write text to temporary file for training
        # (HF tokenizers expects file paths)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(text)
            temp_file = f.name
        
        # Train the tokenizer
        self.tokenizer.train([temp_file], trainer)
        
        # Clean up temp file
        import os
        os.remove(temp_file)
        
        # Set ByteLevel decoder to properly decode
        self.tokenizer.decoder = ByteLevelDecoder()
        
        self._vocab_size = self.tokenizer.get_vocab_size()
        print(f"BPETokenizer: Trained vocabulary with {self._vocab_size} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs, chunking large texts to avoid Rust panics"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        # HuggingFace tokenizers panics on very large strings (>~1GB).
        # Encode in chunks, splitting on newlines to avoid breaking tokens.
        CHUNK_SIZE = 100_000_000  # 100 MB per chunk
        if len(text) <= CHUNK_SIZE:
            return self.tokenizer.encode(text).ids
        all_ids = []
        pos = 0
        chunk_num = 0
        while pos < len(text):
            end = min(pos + CHUNK_SIZE, len(text))
            # Try to split on a newline to avoid breaking mid-token
            if end < len(text):
                newline_pos = text.rfind('\n', pos, end)
                if newline_pos > pos:
                    end = newline_pos + 1
            all_ids.extend(self.tokenizer.encode(text[pos:end]).ids)
            chunk_num += 1
            print(f"  Chunk {chunk_num}: {pos:,} / {len(text):,} characters ({pos*100//len(text)}%)")
            pos = end
        print(f"  Encoding complete: {len(all_ids):,} tokens from {chunk_num} chunks")
        return all_ids
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.decode(tokens)
    
    def save(self, path: str) -> None:
        """Save trained BPE tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained")
        
        # Save the HF tokenizer to a JSON file
        json_path = path.replace('.pkl', '.json')
        self.tokenizer.save(json_path)
        
        # Also save metadata in pickle format for consistency
        data = {
            'tokenizer_type': 'bpe',
            'vocab_size': self._vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'json_path': json_path
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved BPE tokenizer to {path} and {json_path}")
    
    def load(self, path: str) -> None:
        """Load trained BPE tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        assert data['tokenizer_type'] == 'bpe', "Not a BPE tokenizer file"
        
        self.target_vocab_size = data['target_vocab_size']
        self._vocab_size = data['vocab_size']
        
        # Compute json_path from pickle path (don't use stored path which may be stale)
        json_path = path.replace('.pkl', '.json')
        self.tokenizer = HFTokenizer.from_file(json_path)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def tokenizer_type(self) -> str:
        return 'bpe'


class WordPieceTokenizer(Tokenizer):
    """
    WordPiece (BERT-style) tokenizer using HuggingFace tokenizers library.

    Contrast with BPETokenizer (GPT-2 style): pre-tokenization splits on
    whitespace, so tokens carry NO glued-on leading space. Word-initial
    pieces are bare ('the' is just 'the'); continuation pieces carry a
    '##' prefix ('playing' -> 'play', '##ing'). The word boundary lives
    in the '##' convention instead of in a space character, and a common
    word occupies ONE vocab slot instead of the two ('the' / ' the')
    the byte-level style tends to spend.

    Two whitespace caveats, both consequences of the Whitespace
    pre-tokenizer discarding the whitespace itself:
      - Newlines would vanish entirely (the model would never see
        paragraph structure), so encode() maps '\\n' to a [NL] special
        token and decode() maps it back. Line structure round-trips.
      - Runs of spaces/tabs collapse to single spaces on decode, and
        punctuation spacing is reconstructed by the standard BERT
        cleanup rules (' .' -> '.', \"do n't\" -> \"don't\", ...).
        Decoding is therefore NOT byte-lossless, unlike the BPE class.
    """

    NEWLINE_TOKEN = '[NL]'

    def __init__(self, vocab_size: int = 32000):
        self.target_vocab_size = vocab_size
        self.tokenizer: HFTokenizer = None
        self._vocab_size = 0

    def train(self, text: str) -> None:
        """Train WordPiece vocabulary on text (newlines intact — the
        trainer reads line-by-line; [NL] gets its slot as a special)."""
        print(f"WordPieceTokenizer: Training vocabulary of {self.target_vocab_size} tokens...")
        print(f"Training on {len(text):,} characters...")

        self.tokenizer = HFTokenizer(WordPiece(unk_token="[UNK]",
                                               max_input_chars_per_word=100))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=["[UNK]", self.NEWLINE_TOKEN, "<|endoftext|>"],
            continuing_subword_prefix='##',
            show_progress=True
        )

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(text)
            temp_file = f.name

        self.tokenizer.train([temp_file], trainer)

        import os
        os.remove(temp_file)

        self.tokenizer.decoder = WordPieceDecoder(prefix='##', cleanup=True)

        self._vocab_size = self.tokenizer.get_vocab_size()
        print(f"WordPieceTokenizer: Trained vocabulary with {self._vocab_size} tokens")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs. '\\n' becomes the [NL] special token
        (special tokens are extracted before pre-tokenization, so the
        replacement survives even glued to adjacent words). Chunked for
        very large texts, same as BPETokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        text = text.replace('\n', self.NEWLINE_TOKEN)
        CHUNK_SIZE = 100_000_000  # 100 MB per chunk
        if len(text) <= CHUNK_SIZE:
            return self.tokenizer.encode(text).ids
        all_ids = []
        pos = 0
        chunk_num = 0
        while pos < len(text):
            end = min(pos + CHUNK_SIZE, len(text))
            # Split on an [NL] boundary so no word (or the [NL] marker
            # itself) is cut mid-token.
            if end < len(text):
                nl_pos = text.rfind(self.NEWLINE_TOKEN, pos, end)
                if nl_pos > pos:
                    end = nl_pos + len(self.NEWLINE_TOKEN)
            all_ids.extend(self.tokenizer.encode(text[pos:end]).ids)
            chunk_num += 1
            print(f"  Chunk {chunk_num}: {pos:,} / {len(text):,} characters ({pos*100//len(text)}%)")
            pos = end
        print(f"  Encoding complete: {len(all_ids):,} tokens from {chunk_num} chunks")
        return all_ids

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text, restoring '\\n' from [NL]."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        # skip_special_tokens=False so [NL] and <|endoftext|> survive
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        nl = self.NEWLINE_TOKEN
        # The decoder joins pre-tokens with single spaces; strip the
        # spaces it put around each [NL], then drop the marker itself.
        text = text.replace(f' {nl} ', '\n').replace(f'{nl} ', '\n')
        text = text.replace(f' {nl}', '\n').replace(nl, '\n')
        # The rust decoder's cleanup handles ' .'-style punctuation but
        # not apostrophes; apply the remaining BERT cleanup rule so
        # contractions and possessives rejoin ("Don ' t" -> "Don't").
        # Known cosmetic cost: an opening single-quote followed by a
        # word also glues ("said, ' Stop" -> "said,'Stop") — rare in
        # this corpus, which uses double quotes for dialogue.
        text = text.replace(" ' ", "'")
        return text

    def save(self, path: str) -> None:
        """Save trained WordPiece tokenizer (same .pkl + .json pair
        convention as BPETokenizer)."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained")

        json_path = path.replace('.pkl', '.json')
        self.tokenizer.save(json_path)

        data = {
            'tokenizer_type': 'wordpiece',
            'vocab_size': self._vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'json_path': json_path
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved WordPiece tokenizer to {path} and {json_path}")

    def load(self, path: str) -> None:
        """Load trained WordPiece tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        assert data['tokenizer_type'] == 'wordpiece', "Not a WordPiece tokenizer file"

        self.target_vocab_size = data['target_vocab_size']
        self._vocab_size = data['vocab_size']

        json_path = path.replace('.pkl', '.json')
        self.tokenizer = HFTokenizer.from_file(json_path)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def tokenizer_type(self) -> str:
        return 'wordpiece'


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a tokenizer from file.
    Automatically detects type and returns appropriate tokenizer instance.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    tokenizer_type = data['tokenizer_type']
    
    if tokenizer_type == 'char':
        tokenizer = CharTokenizer()
    elif tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(vocab_size=data['target_vocab_size'])
    elif tokenizer_type == 'wordpiece':
        tokenizer = WordPieceTokenizer(vocab_size=data['target_vocab_size'])
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    tokenizer.load(path)
    return tokenizer


# Convenience function for backward compatibility
def create_char_tokenizer(text: str) -> CharTokenizer:
    """Create and train a character tokenizer on text"""
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    return tokenizer


def create_bpe_tokenizer(text: str, vocab_size: int = 16384) -> BPETokenizer:
    """Create and train a BPE tokenizer on text"""
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    return tokenizer
