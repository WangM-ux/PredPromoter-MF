## PredPromoter-MF(2L)



### PredPromoter-MF(2L): A novel approach of promoter prediction based on multi-source feature fusion and deep forest.
This study proposes a novel two-layer predictor, PredPromoter-MF(2L), based on multi-source feature fuses and ensemble learning. PredPromoter-MF(2L) is developed based on various deep features learned by a pre-trained convolutional neural network model and fuse them with the two sequence-derived features. Feature selection based XGBoost is applied to reduce the dimensions of fuse features, and a cascade deep forest model is trained on the selected feature subset for promoter prediction.

***



### Requirements

The following python packages are required when building the stacking model.

Python 3 >= 3.7

keras >= 2.3.1

Sklearn >= 0.0 

tensorflow >= 2.0.0

pandas >= 1.1.2

***



### Running PredPromoter-MF(2L)

open cmd in Windows or terminal in Linux, then cd to the PredPromoter-MF(2L) folder which contains main.py.
</br>`python main.py --fasta [predicting data in fasta format]`</br>  </br>**Example:**
</br>`python main.py --fasta .\dataset\test`</br>  

***



### Announcements

* In order to obtain the prediction results, please be sure to enter the file in fasta format. If you don't know the format of the fasta file, please download the test file to check.


### References
* Chen Z, Zhao P, Li F, Marquez-Lago TT, Leier A, Revote J, Zhu Y, Powell DR, Akutsu T, Webb GI et al: iLearn: an integrated platform and meta-learner for feature engineering, machine-learning analysis and modeling of DNA, RNA and protein sequence data. *Briefings in bioinformatics* 2020, 21(3):1047-1057.
* Zhou ZH, Feng J: Deep Forest. *National Science Review* 2019, 6(1):74-86.
