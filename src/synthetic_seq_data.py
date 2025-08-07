import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from typing import List, Optional
import random


def sort_fn(sequence: torch.Tensor) -> torch.Tensor:
    """
    Sort the input sequence in ascending order.
    
    Args:
        sequence: A tensor of integers
    
    Returns:
        Sorted tensor
        
    Examples:
        >>> sort_fn(torch.tensor([4, 12, 3, 7]))  # tensor([3, 4, 7, 12])
    """
    return torch.sort(sequence)[0]


def reverse_fn(sequence: torch.Tensor) -> torch.Tensor:
    """
    Reverse the input sequence.
    
    Args:
        sequence: A tensor of integers
    
    Returns:
        Reversed tensor
        
    Examples:
        >>> reverse_fn(torch.tensor([4, 12, 3, 7]))  # tensor([7, 3, 12, 4])
    """
    return torch.flip(sequence, dims=[0])


def count_fn(sequence: torch.Tensor) -> torch.Tensor:
    """
    COUNT task: Given [min_val, max_val], count from min_val to max_val inclusive.
    
    Args:
        sequence: A tensor of length 2: [min_val, max_val]
    
    Returns:
        Tensor counting from min_val to max_val inclusive
        
    Examples:
        >>> count_fn(torch.tensor([2, 5]))  # tensor([2, 3, 4, 5])
        >>> count_fn(torch.tensor([7, 9]))  # tensor([7, 8, 9])
        >>> count_fn(torch.tensor([3, 3]))  # tensor([3])
    """
    if len(sequence) != 2:
        raise ValueError(f"COUNT task expects sequence of length 2, got {len(sequence)}")
    
    min_val, max_val = sequence[0].item(), sequence[1].item()
    
    # Handle edge case where min > max
    if min_val > max_val:
        return torch.tensor([], dtype=sequence.dtype)
    
    # Count from min_val to max_val inclusive
    return torch.arange(min_val, max_val + 1, dtype=sequence.dtype)


class SyntheticSequenceDataset(Dataset):
    """
    Dataset for synthetic sequence-to-sequence tasks with variable length inputs.
    
    This dataset generates sequences of the form: [input] + [sep_token] + [output]
    For next-token prediction training, where the model learns to predict the continuation.
    
    Example with SORT task:
    Input sequence: [4, 12, 3, 7]
    Output sequence: [3, 4, 7, 12] 
    Full sequence: [4, 12, 3, 7, 102, 3, 4, 7, 12]  (102 is separator '>')
    
    The model will be trained on next-token prediction on this full sequence.
    """
    
    def __init__(
        self,
        n_samples: int,
        min_seq_len: int,
        max_seq_len: int,
        vocab_size: int,
        ground_truth_fn: callable,
        sep_token: int = 102,  # Using 102 for '>' separator (assuming vocab_size < 102)
        pad_token: int = 103,  # Using 103 for padding
    ):
        super().__init__()
        self.n_samples = n_samples
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.ground_truth_fn = ground_truth_fn
        self.sep_token = sep_token
        self.pad_token = pad_token

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        # Generate a fresh sequence on-the-fly for each request
        # 1) Sample sequence length uniformly from [min_seq_len, max_seq_len]
        seq_len = torch.randint(self.min_seq_len, self.max_seq_len + 1, (1,)).item()
        
        # 2) Sample random sequence from vocab_size alphabet
        input_seq = torch.randint(0, self.vocab_size, (seq_len,))
        
        # 3) Apply ground truth function to get output sequence
        output_seq = self.ground_truth_fn(input_seq)
        
        # Create full sequence: [input] + [sep_token] + [output]
        # Note: No padding here - that's handled in the collate function
        full_seq = torch.cat([
            input_seq,
            torch.tensor([self.sep_token]),
            output_seq
        ])
        
        return full_seq


class CountSequenceDataset(Dataset):
    """
    Dataset for COUNT task: given [min_val, max_val], generate sequence from min to max.
    
    This dataset generates sequences of the form: [min_val, max_val] + [sep_token] + [min_val, min_val+1, ..., max_val]
    For next-token prediction training, where the model learns to count from min to max.
    
    Example:
    Input: [2, 5]
    Output: [2, 3, 4, 5] 
    Full sequence: [2, 5, 102, 2, 3, 4, 5]  (102 is separator '>')
    """
    
    def __init__(
        self,
        n_samples: int,
        min_range_size: int = 1,  # Minimum size of counting range (max - min)
        max_range_size: int = 10, # Maximum size of counting range (max - min)
        vocab_size: int = 20,
        sep_token: int = 102,
        pad_token: int = 103,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.min_range_size = min_range_size
        self.max_range_size = max_range_size
        self.vocab_size = vocab_size
        self.sep_token = sep_token
        self.pad_token = pad_token

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        # Generate a fresh counting problem on-the-fly
        # 1) Sample range size uniformly from [min_range_size, max_range_size]
        range_size = torch.randint(self.min_range_size, self.max_range_size + 1, (1,)).item()
        
        # 2) Sample min_val ensuring max_val stays within vocab_size
        max_possible_min = self.vocab_size - range_size
        min_val = torch.randint(0, max_possible_min, (1,)).item()
        max_val = min_val + range_size
        
        # 3) Create input sequence [min_val, max_val]
        input_seq = torch.tensor([min_val, max_val])
        
        # 4) Apply count function to get output sequence
        output_seq = count_fn(input_seq)
        
        # Create full sequence: [min_val, max_val] + [sep_token] + [output]
        full_seq = torch.cat([
            input_seq,
            torch.tensor([self.sep_token]),
            output_seq
        ])
        
        return full_seq



class PositionShiftCollate:
    """Collate function class that can be pickled for multiprocessing."""
    
    def __init__(self, max_positions: int = 512, training: bool = True):
        self.max_positions = max_positions
        self.training = training
    
    def __call__(self, batch: List[torch.Tensor]) -> dict:
        return position_shift_collate_fn(batch, self.max_positions, self.training)


def position_shift_collate_fn(batch: List[torch.Tensor], max_positions: int = 512, training: bool = True) -> dict:
    """
    Custom collate function that implements position shifting for length generalization.
    Handles variable-length sequences by padding them to the same length.
    
    This implements the mechanism from the paper: "at train time, we add random offsets 
    to position indices so that all position embeddings are trained. The offsets are 
    sampled uniformly at random in the range [0, N − |x|]"
    
    Args:
        batch: List of variable-length sequences from the dataset
        max_positions: Maximum position embedding size (N in the paper)
        training: Whether we're in training mode (only shift during training)
    
    Returns:
        Dictionary with 'input_ids' and 'position_ids'
    """
    # Find the maximum length in the batch for padding
    max_len = max(len(seq) for seq in batch)
    
    # Get pad token from the first sequence (assumes all sequences use same pad token)
    # We'll use 103 as default pad token if we can't infer it
    pad_token = 103
    
    # Pad all sequences to the same length
    padded_batch = []
    for seq in batch:
        if len(seq) < max_len:
            padded_seq = F.pad(seq, (0, max_len - len(seq)), value=pad_token)
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)
    
    # Stack batch into tensor: (batch_size, seq_len)
    input_ids = torch.stack(padded_batch)
    batch_size, seq_len = input_ids.shape
    
    if training and seq_len < max_positions:
        # Sample random offsets uniformly from [0, N - |x|] for each sequence
        max_offset = max_positions - seq_len
        offsets = torch.randint(0, max_offset + 1, (batch_size, 1))
        
        # Create position IDs starting from the random offset
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len) + offsets
    else:
        # No shifting during evaluation or when seq_len >= max_positions
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    
    return {
        'input_ids': input_ids,
        'position_ids': position_ids
    }


class SyntheticSequenceDataModule(LightningDataModule):
    """
    DataModule for SyntheticSequenceDataset to work with Lightning training.
    """
    
    def __init__(
        self,
        train_dataset: SyntheticSequenceDataset,
        val_dataset: SyntheticSequenceDataset,
        batch_size: int,
        num_workers: int = 0,
        test_dataset: Optional[SyntheticSequenceDataset] = None,
        max_positions: int = 512,
        enable_position_shifting: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset is not None else val_dataset
        self.max_positions = max_positions
        self.enable_position_shifting = enable_position_shifting

    def train_dataloader(self) -> DataLoader:
        # Use position shifting collate function for training
        collate_fn = None
        if self.enable_position_shifting:
            collate_fn = PositionShiftCollate(
                max_positions=self.max_positions, 
                training=True
            )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        # No position shifting during validation
        collate_fn = None
        if self.enable_position_shifting:
            collate_fn = PositionShiftCollate(
                max_positions=self.max_positions, 
                training=False
            )
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        # No position shifting during testing
        collate_fn = None
        if self.enable_position_shifting:
            collate_fn = PositionShiftCollate(
                max_positions=self.max_positions, 
                training=False
            )
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    print("=== SyntheticSequenceDataset Examples ===\n")
    
    # Example 1: SORT task
    print("1. SORT Task Example:")
    sort_dataset = SyntheticSequenceDataset(
        n_samples=3,
        min_seq_len=3,
        max_seq_len=5,
        vocab_size=20,  # Use integers 0-19
        ground_truth_fn=sort_fn,
        sep_token=102,  # '>' separator
        pad_token=103   # padding
    )
    
    for i in range(len(sort_dataset)):
        seq = sort_dataset[i]
        print(f"Sample {i}: {seq.tolist()}")
        
        # Parse the sequence
        sep_pos = (seq == 102).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            sep_idx = sep_pos[0].item()
            input_part = seq[:sep_idx]
            output_part = seq[sep_idx+1:]
            # Remove padding from output
            output_part = output_part[output_part != 103]
            print(f"  Input: {input_part.tolist()}")
            print(f"  Output: {output_part.tolist()}")
            print(f"  Verification - sorted input: {torch.sort(input_part)[0].tolist()}")
        print()
    
    print("=" * 50)
    
    # Example 2: REVERSE task
    print("2. REVERSE Task Example:")
    reverse_dataset = SyntheticSequenceDataset(
        n_samples=3,
        min_seq_len=3,
        max_seq_len=4,
        vocab_size=10,  # Use integers 0-9
        ground_truth_fn=reverse_fn,
        sep_token=102,  # '>' separator
        pad_token=103   # padding
    )
    
    for i in range(len(reverse_dataset)):
        seq = reverse_dataset[i]
        print(f"Sample {i}: {seq.tolist()}")
        
        # Parse the sequence
        sep_pos = (seq == 102).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            sep_idx = sep_pos[0].item()
            input_part = seq[:sep_idx]
            output_part = seq[sep_idx+1:]
            # Remove padding from output
            output_part = output_part[output_part != 103]
            print(f"  Input: {input_part.tolist()}")
            print(f"  Output: {output_part.tolist()}")
            print(f"  Verification - reversed input: {torch.flip(input_part, dims=[0]).tolist()}")
        print()
    
    print("=" * 50)
    print("Key points:")
    print("- Each sequence contains: [input] + [separator_token] + [output]")
    print("- Model will learn next-token prediction on the entire sequence")
    print("- Separator token (102) indicates transition from input to output")
    print("- Padding token (103) is used to make all sequences the same length")
    print("- Variable sequence lengths sampled uniformly from [min_seq_len, max_seq_len]")