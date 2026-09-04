from .cnn_classic import ClassicCNN, ClassicCNNActor, ClassicCritic
from .cnn_attention import AttentionCNN, AttentionCNNActor, AttentionCritic

__all__ = [
    'ClassicCNN', 'ClassicCNNActor', 'ClassicCritic',
    'AttentionCNN', 'AttentionCNNActor', 'AttentionCritic'
]
