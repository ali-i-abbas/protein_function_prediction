# Sequence-based protein function prediction using convolutional neural networks


This is my final project for CAP6545. Please refer to my final report (Sequence_based_protein_function_prediction_using_convolutional_neural_networks___Ali_Abbas.pdf) for more details.

### Motivation: 

Genome sequencing advances in recent years have resulted in huge amounts of protein sequence data. To make use of such data, we first need to map each protein sequence to relevant biological processes, cellular components, or molecular functions. This is known as Gene Ontology (GO) term annotation prediction problem. In other words, the problem is that given a protein sequence data as input, how to predict its GO term annotations. Manual labeling of data is very time consuming so we need to automate this process.

 
### Results: 

Recently, Zuallaert et al., 2019, studied the effects of different encoding strategies on protein function prediction performance using a Convolutional Neural Network (CNN). I reimplemented a reduced scope of their work by selecting only one-hot encoded and ad-hoc trainable embeddings encoding strategies and dropping trigrams. I was unable to get the same results despite using the same parameters but got the same trends of seeing simpler encodings performing better. Next, in order to improve the results, I augmented the unigram one-hot encoded input with six chemical properties of the amino acids and got slightly better results than my first implementation, especially for BP and MF functions, which might be due to BP and MF having stronger correlation to chemical properties of amino acids compared to CC. Then, in order to further improve my results, I used the Tree-structured Parzen estimators (TPE) to search for best hyperparameters, which resulted in better performance but still not matching the original paper’s results. Finally, I used a bottleneck model and was able to match the original paper’s results for CC functions, and get to 1.2% distance from MF results. The final model’s improved results could be due to introducing an information bottleneck before the last layer which causes the model to learn to generalize from limited data.

### Dataset:

Dataset can be downloaded from http://deepgo.bio2vec.net/data/deepgo/data.tar.gz. Extract and copy the following files to data folder:

+ data/go.obo
+ data/train/bp.pkl
+ data/train/cc.pkl
+ data/train/mf.pkl
+ data/train/train-bp.pkl
+ data/train/train-cc.pkl
+ data/train/train-mf.pkl
+ data/train/test-bp.pkl
+ data/train/test-cc.pkl
+ data/train/test-mf.pkl

### Models:

All the trained models are saved in models folder with the following naming convention:

<model_name>_<encoding>_<subontology>_e<nb_epoch>_b<batch_size>_n<gram_len>_v<embedding_size>_r<run>.h5

+ encoding can be 'oh' for one-hot encoding or 'ad' for ad-hoc trainable embedding
+ subontology: ['cc', 'mf', 'bp']
+ gram_len: n for n-gram
+ embedding_size: size of embedding vector v
+ run: run number

### Code:

- baseline.py runs my implementation of the baseline model. You can change the main function default parameters. model_name is baseline.
- baseline_experiments.py runs the experiments of the baseline model and the results are saved in results_baseline_<number of epochs>.csv. model_name is baseline_exp.
- chemical.py runs my implementation of the baseline model with augmented chemical properties of the amino acids. You can change the main function default parameters. model_name is chemical.
- chemical_experiments.py runs the experiments of the baseline model with augmented chemical properties of the amino acids and the results are saved in results_chemical_<number of epochs>.csv. model_name is chemical_exp.
- tpe_model.py runs my implementation of the TPE model. You can change the main function default parameters. model_name is tpe.
- tpe_model_experiments.py runs the experiments of the TPE model and the results are saved in results_tpe_<number of epochs>.csv. model_name is tpe_exp.
- bottleneck_model.py runs my implementation of the bottleneck model. You can change the main function default parameters. model_name is bottleneck.
- bottleneck_model_experiments.py runs the experiments of the bottleneck model and the results are saved in results_bottleneck_<number of epochs>.csv. model_name is bottleneck_exp.


- tpe_params.py runs the TPE algorithm to search for hyperparameters. the best model is saved in params folder. All the parameters are saved in parameters.csv and parameters.pkl. The trials are saved in hyperopt_trials.pkl, which can be used to resume the search. The best found parameters are saved in best_param.log.

- Final.R is R script used to generate graphs and do statistics on results.




### References

Zuallaert, J., Pan, X., Saeys, Y., Wang, X., and De Neve, W. (2019).  Investigatingthe biological relevance in trained embedding representations of protein sequences.InWorkshop on Computational Biology at the 36th International Conference onMachine Learning (ICML 2019).
