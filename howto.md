Bien que Tensorflow fonctionne sur Windows, ce projet n'a été testé qu'avec Linux.

# Github

### Code original
https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks
### Code modifé
https://github.com/Dref360/tensorflow-value-iteration-networks


## Requis
* Python 3
* OpenCV 3.1

## Installation
```python
# Tensorflow CPU
pip install tensorflow
# Tensorflow GPU
pip install tensorflow-gpu
```

Dans le README du répertoire, il y a un lien vers le dataset à télécharger.
Par contre, le dataset 8x8 est dans le dossier */data*.

### Entrainer
```
python train.py
```

Il y a beaucoup d'options, mais la batch normalization est obligatoire.

Options:
```
usage: train.py [-h] [--input INPUT] [--imsize IMSIZE] [--lr LR]
                [--epochs EPOCHS] [--k K] [--ch_i CH_I] [--ch_h CH_H]
                [--ch_q CH_Q] [--batchsize BATCHSIZE]
                [--statebatchsize STATEBATCHSIZE]
                [--untied_weights [UNTIED_WEIGHTS]] [--nountied_weights]
                [--show [SHOW]] [--noshow] [--display_step DISPLAY_STEP]
                [--log [LOG]] [--nolog] [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to data
  --imsize IMSIZE       Size of input image
  --lr LR               Learning rate for RMSProp
  --epochs EPOCHS       Maximum epochs to train for
  --k K                 Number of value iterations
  --ch_i CH_I           Channels in input layer
  --ch_h CH_H           Channels in initial hidden layer
  --ch_q CH_Q           Channels in q layer (~actions)
  --batchsize BATCHSIZE
                        Batch size
  --statebatchsize STATEBATCHSIZE
                        Number of state inputs for each sample (real number,
                        technically is k+1)
  --untied_weights [UNTIED_WEIGHTS]
                        Untie weights of VI network
  --nountied_weights
  --show [SHOW]         Shows the value map at the end (SLOW the training a
                        lot)
  --noshow
  --display_step DISPLAY_STEP
                        Print summary output every n epochs
  --log [LOG]           Enable for tensorboard summary
  --nolog
  --logdir LOGDIR       Directory to store tensorboard summary
```

Les valeurs pas défauts sont celles recommandées par l'article.

```
#Exemple de sortie sur 8x8
#...
26 |  0.0335515 | 0.00968364 |    16.8366
27 |  0.0331753 |  0.0099537 |    16.1874
28 |  0.0327328 | 0.00958076 |    15.4125
29 |  0.0320567 | 0.00928498 |    15.7143
Finished training!
Accuracy: 98.59999995678663%
Model Saved!

Process finished with exit code 0
```
### Test

Pour simuler l'éxécution de l'algorithme pour un vrai problème, j'ai créé le fichier *test.py*.

Le fichier va charger un modèle déjà entrainé.
Il va ensuite éxécuter le modèle à chaque pas pour choisir l'action afin d'atteindre le but.

```
python test.py
```

Les mêmes options sont disponibles que pour *train.py*. Pour chaque entrée du test set, une image apparaitra avec le nombre de bonne prédiction comme titre.

### Voir la **Test Accuracy** (Comme dans les diapositives)

Il suffit de démarrer tensorboard, si la config *logdir* n'a pas été altéré :

```tensorboard --logdir /tmp/vintf/```
