from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)
print('example of dataset elements', books["train"][0])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# the following 2 hyperparameters are task-specific
max_en_length = 256
max_fr_length = 256

# Suppose we have the following 2 training examples:
input_sequence_en_1 = "Welcome to NYC"
input_sequence_en_2 = "HuggingFace is a company"
input_sequence_fr_1= "Bienvenue à NYC"
output_sequence_fr_2 = "HuggingFace est une entreprise"

# encode the inputs
task_prefix_enfr = "translate English to French: "
task_prefix_fren = "translate French to English: "
input_sequences_en = [input_sequence_en_1, input_sequence_en_2]
input_sequences_fr = [input_sequence_fr_1, output_sequence_fr_2]

encoding_en = tokenizer(
    [task_prefix_enfr + sequence for sequence in input_sequences_en],
    padding="longest",
    max_length=max_en_length,
    truncation=True,
    return_tensors="pt",
)
encoding_fr = tokenizer(
    [task_prefix_fren + sequence for sequence in input_sequences_fr],
    padding="longest",
    max_length=max_fr_length,
    truncation=True,
    return_tensors="pt",
)
decoding_fr = tokenizer(
    input_sequences_fr,
    padding="longest",
    max_length=max_fr_length,
    truncation=True,
    return_tensors="pt",
)
decoding_en = tokenizer(
    input_sequences_en,
    padding="longest",
    max_length=max_en_length,
    truncation=True,
    return_tensors="pt",
)

input_ids_en, attention_mask_en = encoding_en.input_ids, encoding_en.attention_mask
input_ids_fr, attention_mask_fr = encoding_fr.input_ids, encoding_fr.attention_mask
labels_en = decoding_en.input_ids
labels_fr = decoding_fr.input_ids


# replace padding token id's of the labels by -100 so it's ignored by the loss
# labels[labels == tokenizer.pad_token_id] = -100

# autoreg generation
# output_enfr = model.generate(input_ids=input_ids_en, attention_mask=attention_mask_en, max_length=max_en_length)
# tokenizer.decode(output_enfr[1])


# forward pass
output_enfr = model(input_ids=input_ids_en, attention_mask=attention_mask_en, labels=labels_fr)
output_fren = model(input_ids=input_ids_fr, attention_mask=attention_mask_fr, labels=labels_en)

loss_enfr = output_enfr.loss
loss_fren = output_fren.loss
print('loss_enfr:', loss_enfr.item())
print('loss_fren:', loss_fren.item())
print('input sentence 0: ', input_sequence_en_1, ' \n',
'output sentence 0 :', tokenizer.decode(output_enfr.logits.argmax(-1)[0], skip_special_tokens=True), ' \n',
'input sentence 1: ', input_sequence_en_2, ' \n',
'output sentence 1 :', tokenizer.decode(output_enfr.logits.argmax(-1)[1], skip_special_tokens=True))
print('input sentence 0: ', input_sequence_fr_1, ' \n',
'output sentence 0 :', tokenizer.decode(output_fren.logits.argmax(-1)[0], skip_special_tokens=True), ' \n',
'input sentence 1: ', output_sequence_fr_2, ' \n',
'output sentence 1 :', tokenizer.decode(output_fren.logits.argmax(-1)[1], skip_special_tokens=True))




def UnwrappedT5Test():
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    unwrapped_model = UnwrappedPretrainedT5(base=False)
    model = unwrapped_model['model']
    vector_model = unwrapped_model['vector_model']
    input_discretizer = unwrapped_model['discretizer_enc']
    output_discretizer = unwrapped_model['discretizer_dec']

    input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

    # the forward function automatically creates the correct decoder_input_ids
    logits = model(input_ids=input_ids, labels=labels).logits
    print(tokenizer.batch_decode(logits.argmax(dim=-1)))
    
    labels_with_pad = torch.cat((torch.ones((labels.shape[0], 1), dtype=torch.long).to(labels.device)*tokenizer.pad_token_id, labels), dim=1)
    # using vectorized model
    input_vector_embeddings = input_discretizer.encoder_embedding_from_id(input_ids)
    input_attention_mask = torch.ones_like(input_ids)
    output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels_with_pad)
    output_attention_mask = torch.ones_like(labels_with_pad)
    # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels)
    # output_attention_mask = torch.ones_like(labels)
    output_vector_model = vector_model(inputs_embeds=input_vector_embeddings, attention_mask=input_attention_mask,
                                        decoder_inputs_embeds=output_vector_embeddings, decoder_attention_mask=output_attention_mask
                                        ,use_cache= None,)
    discretized_output = output_discretizer(output_vector_model['last_hidden_state'])
    print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))




