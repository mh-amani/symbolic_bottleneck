### Install dependecy and the library
```bash
conda create -n blocks python=3.11
conda activate blocks
pip install -r requirements.txt
pip install -e .
```

### How to use the library
You can use models in blocks/models directory directly. They are ready to use.
Otherwise you can use the blocks/autoregressive_wrapper and blocks/modules/discrete_bottleneck to create your own models with autoregressive backpropagation via straight-through estimator.

You can then connect two of these models together, e.g. bart_french_english to bart_english_german via blocks/blocks_connectors.


### How to add models
You can add models in blocks/models directory. They should be a subclass of torch.nn.Module like the examples in blocks/models directory.
create a file for your model. Implement a model unwrapper, to take away any language modeling mask and word embedding, so we're left with pure vector transformer/other model. you can use the unwrapper in blocks/models/model_unwrapper/...

Usually models have funny ways of inner working. in T5 you should append a EOS token (word_embed[2]) and in bart you should append a EOS token (word_embed[2]) and a language id (for translation). figure this out and make sure your model is working as you want it by writing a test in examples/test_outputs/your_model.py.

Then you should wrap your model in an autoreg_wrapper (/blocks/auto_reg_wrapper.py). This model should autoregressively generate output in a way that is differentiable. 
We need to both make sure that gradients look fine, and that the model is trainable.

