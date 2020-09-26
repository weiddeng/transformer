import os
import numpy as np
import trax
from trax import layers as tl
from trax.supervised import training

# 1. Why vocab_size=33300? 2. mode='train' good for eval?
model = trax.models.TransformerEncoder(
    vocab_size=33300, n_classes=2, d_model=512, d_ff=2048, n_heads=2, n_layers=2, max_len=2048, mode='train'
)
print(model)

train_stream = trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True)()
eval_stream = trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=False)()
print(next(train_stream))

data_pipeline = trax.data.Serial(
    trax.data.Tokenize(vocab_file='en_8k.subword', keys=[0]),
    trax.data.Shuffle(),
    trax.data.FilterByLength(max_length=2048, length_keys=[0]),
    trax.data.BucketByLength(boundaries=[  32, 128, 512, 2048],
                             batch_sizes=[512, 128,  32,    8, 1],
                             length_keys=[0]),
    trax.data.AddLossWeights()
)
train_batches_stream = data_pipeline(train_stream)
eval_batches_stream = data_pipeline(eval_stream)
example_batch = next(train_batches_stream)
print(f'shapes = {[x.shape for x in example_batch]}')

# Training task.
train_task = training.TrainTask(
    labeled_data=train_batches_stream,
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=5,
)

# Evaluaton task.
eval_task = training.EvalTask(
    labeled_data=eval_batches_stream,
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    n_eval_batches=20  # For less variance in eval numbers.
)

# Training loop saves checkpoints to output_dir.
output_dir = os.path.expanduser('~/output_dir/')
training_loop = training.Loop(model,
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)

# Run 50 steps (batches). Takes about 12 mins.
training_loop.run(50)

example_input = next(eval_batches_stream)[0][0]
example_input_str = trax.data.detokenize(example_input, vocab_file='en_8k.subword')
print(f'example input_str: {example_input_str}')
sentiment_log_probs = model(example_input[None, :])  # Add batch dimension.
print(f'Model returned sentiment probabilities: {np.exp(sentiment_log_probs)}')