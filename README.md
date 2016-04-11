# Problem description

The description of the problem can be found here: https://www.kaggle.com/c/dato-native

The task is to predict which web pages served by StumbleUpon company are sponsored.

When native advertising is done right, users aren't desperately scanning an ad for a hidden "x". In fact, they don't even know they're viewing one. To pull this off, native ads need to be just as interesting, fun, and informative as the unpaid content on a site.

If media companies can better identify poorly designed native ads, they can keep them off your feed and out of your user experience. 


# Getting the data

The competition is over, but we can still download the trianing and testing data here: https://www.kaggle.com/c/dato-native/data

You would have to register at Kaggle.com to download the data! It is just one click away.

Note, you would need only need labeled data, which we are going to use for training and cross validation:

##### train\_v2.csv
##### {0,1,2,3,4}.zip

## File descriptions

{0,1,2,3,4}.zip - are all HTML files. 
Files listed in train\_v2.csv are training files:

```bash
file - the raw file name
sponsored - 0: organic content; 1: sponsored content label
```


We are going to use a fraction of fake data for exploratory analysis with Spark in the iPython notebook. test with the full dataset will be illustrated during the live demo.
