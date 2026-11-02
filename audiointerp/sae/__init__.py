"""TopK sparse autoencoders on pooled layer activations, following Mariotte et al. 2025
("Sparse Autoencoders Make Audio Foundation Models More Explainable", arXiv 2509.24793).

Run order: train -> (audiointerp.egemaps) -> disentanglement -> selective_units, factor_units.
"""
