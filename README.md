# PairBuddy-Classifier

An application of an SVM machine learning model to classify three facets of pair programming

## Research Paper

This code accompanies the following published research at VL/HCC 2019:

https://ieeexplore.ieee.org/document/9127250

## How to Run

Runner.py is the main class

You must get a credential.json from this website and put it in the project directory
https://developers.google.com/sheets/api/quickstart/go

You will be prompeted to link QuickStart with you google account so it has access to the document

Settings.py reads the settings.json file that tells the program what columns to use, what classifier to use and other settings. It is used in both SheetToDataFrame.py and Classifier.py

SheetToDataFrame.py calls the Google Sheets API and retrieves all of the studies and puts it into a list of DataFrames (one for each study). It also converts all time columns to floats.

Classifier.py contains both preprocessing functions and the classifier functions. At the bottom of Classifier.py is a run() function that calls preprocessing and classifier functions. All preprocessing functions have 2 copies, one for if all the studies are in a single DataFrame and another for if the studies are in a list of DataFrames individually. I don't actually use any of the DataFrame only variants, but I also don't want to delete all that work. You can delete them if you'd like. 
