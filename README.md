## Datasets
### For authors classification
https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution <br>
In total it has 53678 rows and 50 different authors <br>
We simplified the task to 2990 rows and 4 different authors

### For imdb sentiment classification
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews <br>
In total it has 49582 rows and 2 label: positive and negative <br>
We simplified the task to 3000 rows

## Independent models
2 separately trained models

## Multitask model
One model with 2 heads: author classification and imdb sentiment

## Results
### Authors
<pre>
  48.   0.   0.   0.   <br>
  1.   20.   0.   0.   <br>
  1.   0.   4.   1.   <br>
  2.   8.   0.   64.   <br>
  Accuracy: 0.91
</pre>  
### Imdb
<pre>
  70.   7.   <br>
  4.   69.   <br>
  Accuracy: 0.93
</pre>
