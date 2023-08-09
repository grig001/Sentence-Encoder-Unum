# Sentence-Encoder-Unum

In file My_dataset_script.py I create a my_data.csv file and put a 20 million sentence from bookcorpus dataset in it (all dataset(74 million) is too much), then in file tokenization_and_mask_in_train_loop.py I read the created file and datas in it.

<h1 align="center">Bert pre-training with retrieval purposes</h1>
<h3 align="center">
The project is the LM pre-training pipeline with retrieval purposes. <br/>
Here you can find:<br/>
</h3>
<br/>

* [x] Evaluations tasks for retrieval (MRPC, STS-b). Both from GLUE benchmark
* [x] Dataset preparation scripts
* [x] Pre-training using Masked LM task on Wikipedia data.
* [x] WanDB logging
* [x] MultiGPU training code
* [x] Checkpointing
* [ ] Fine-tune code using contrastive learning.
* [ ] Results and checkpoints reported
