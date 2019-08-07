Should figure out how to cut short the GPU memory.
Part4 collapses when loading dataset 9 (out of memory) - out of 17.

It used to work:
    - Transfer all models to float instead of double.
    - Try to put torch.cuda.empty_cache in each round.
