rna_k_mer_tokenizer.py: creates tokenizer .json file by reading k-mer pretraining data

bert-rna-model.json: Find an online example for Bert configuration and modified it. Reduced number of layers and vocabulary size. Added num_labels

bert-rna-6-mer-tokenizer.json: Output of run_k_mer_tokenizer.py.

make_k_mers.py: turns nucleotide sequence into given k-mer sequences.

run_mlm.py: masked language model pretraining. Modified to pretrain from scratch and to read sequence data. Default values are updated for our purpose.

fintune.py: finetunes pretrained model with family Classification task

plot_metrics.py: Gets checkpoint directory and plots loss, accuracy

plot_dataset.py: Used for dataset length distribution and size.




conda create -n CS230 python=3.10
pip install -r requirements.txt

python run_mlm.py --output_dir ./out_mlm
python run_mlm.py --output_dir ./out_mlm --resume ./out_mlm/chekpoint-XXXX

python run_cls.py --output_dir ./out_cls --model_name_or_path ./out_mlm/
python run_cls.py --output_dir ./out_cls --resume ./out_cls/checkpoint-XXXX
