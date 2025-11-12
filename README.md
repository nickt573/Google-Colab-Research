# Google-Colab-Research
Samples of introductory research practices using Google Colab under the direction of Binghamton University researcher Dr. Kenneth Chiu.

image classifier is a Colab notebook paired with a Python script built on Python's PyTorch. The model reads a Kagglehub chest x-ray dataset, training, validation, and recording data for each epoch. The model has inference capabilities and reports training loss over time. The model with its current configuration has been seen with up to ~85% validation accuracy.

transformer is a language transformer that trains and translates based on data from Hugging Face. The Colab notebook must be configured to properly save and load generated weights based on the desired langauge as described in the first cell of the notebook. The majority of the code has been adapted from Umar Jamil (https://github.com/hkproj/pytorch-transformer) and significanty modified and simplified for the purposes of my research. 
