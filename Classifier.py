import time
from typing import List

import contractions
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter
from Settings import Settings


class Classifier:
    """
    Contains both the preprocessing steps for features and the classifiers
    All preprocessing functions are in pairs: one that applies to a list of DataFrames and another to a single DataFrame
    """
    def __init__(self, settings: Settings):

        # Make sure these are already downloaded
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        # Keep a reference to the settings
        self.settings = settings

        # These are lists of stop words that I remove from nltk's stopwords
        # If you want to exclude some of these, just comment them out in the same way that self.object_words is
        self.people_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                             'you', 'your', 'yours', 'yourself', 'yourselves',
                             'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself']
        self.they_words = ['they', 'them', 'their', 'theirs', 'themselves']
        #self.object_words = ['it', 'its', 'itself']
        self.question_words = ["what", "which", "how", "who", "whom", "when", "where", "why"]
        self.misc_words = ['should', 'can', 'same', 'here', 'there', 'again', 'over', 'under', 'off', 'on', 'in', 'out', 'up',
                            'down', 'before', 'after', 'above', 'below', 'between', 'against', 'not', 'no', 'but', 'because',
                            'into', 'each', 'few', 'more', 'most', 'other', 'only', 'own', 'than']
        self.sw = stopwords.words('english')
        [self.sw.remove(x) for x in self.people_words] #Chck on this
        [self.sw.remove(x) for x in self.they_words]  # Chck on this
        #[self.sw.remove(x) for x in self.object_words]  # Chck on this
        [self.sw.remove(x) for x in self.question_words]  # Chck on this
        [self.sw.remove(x) for x in self.misc_words]  # Chck on this
        ', '.join(self.sw)

        # These are all lists of words that are combined to one word
        self.number_list = ['zero', '0', 'one', '1', 'two', '2', 'three', '3', 'four', '4', 'five', '5', 'six', '6',
                            'seven', '7', 'eight', '8', 'nine', '9', 'ten', '10']
        self.uh_list = ['uh', 'um', 'hm', 'hmm', 'eh', 'ehh', 'uhm']
        self.yes_list = ['yes', 'ya', 'yep', 'yuh', 'yeah']
        self.no_list = ['no', 'nope', 'nah']

        self.label_list = list()


    def __replace_contractions(self, text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    def __remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in self.sw:
                new_words.append(word)
        return new_words

    def __lemmatize(self, words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for (word, tag) in nltk.pos_tag(words):
            if tag == 'NNS':
                lemmas.append(lemmatizer.lemmatize(word, 'n'))
            elif tag.startswith('VB'):
                lemmas.append(lemmatizer.lemmatize(word, 'v'))
            elif tag in ['RBR', 'RBS']:
                lemmas.append(lemmatizer.lemmatize(word, 'r'))
            elif tag in ['JJR', 'JJS']:
                lemmas.append(lemmatizer.lemmatize(word, 'a'))
            else:
                lemmas.append(word)
        return lemmas

    def __to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def __remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def __combine_words(self, words_list: List[str], replacement: str, words: List[str]) -> List[str]:
        """
        Replaces any word from 'words' with the 'replacement' if it is from the 'words_list'
        :param words_list: list of words to replace
        :param replacement: the word to replace them with
        :param words: the list of words to change
        :return: a new potentially modified list of words
        """
        new_words = []
        for word in words:
            if word in words_list:
                new_words.append(replacement)
            else:
                new_words.append(word)
        return new_words

    def __normalize(self, words):
        words = self.__to_lowercase(words)
        words = self.__lemmatize(words)
        words = self.__remove_stopwords(words)
        words = self.__remove_punctuation(words)
        return words

    def preprocess(self, sentence):
        sentence = self.__replace_contractions(sentence)
        words = nltk.word_tokenize(sentence)
        words = self.__normalize(words)
        words = self.__combine_words(self.number_list, "NUMBER", words)
        words = self.__combine_words(self.uh_list, "UH", words)
        words = self.__combine_words(self.yes_list, "YES", words)
        words = self.__combine_words(self.no_list, "NO", words)
        return str(' '.join(words))

    def preprocessCategoricalWord(self, tag):
        """Removes all spaces, lowercase, remove punctuation"""
        tag = tag.replace(" ", "")
        tag = tag.lower()
        tag = re.sub(r'[^\w\s]', '', tag)
        return tag

    def preprocessDialogueInListOfDataFrames(self, list_of_dfs):
        """
        Applies the preprocess function to the dialogue column of each DataFrame in a list
        Replaces any resulting nan entries with ''
        :param list_of_dfs: list of DataFrames each with a dialgoue column
        """
        for i in range(len(list_of_dfs)):
            list_of_dfs[i][self.settings.dialogue_column] = list_of_dfs[i][self.settings.dialogue_column].apply(self.preprocess)
            list_of_dfs[i][self.settings.dialogue_column] = list_of_dfs[i][self.settings.dialogue_column].replace(np.nan, '', regex=True)

    def preprocessDialogueInDataFrame(self, df: pd.DataFrame):
        """
        Applies the preprocess function to the dialogue column of a DataFrame
        Replaces any resulting nan entries with ''
        :param df: DataFrame with a dialgoue column
        """
        df[self.settings.dialogue_column] = df[self.settings.dialogue_column].apply(self.preprocess)
        df[self.settings.dialogue_column] = df[self.settings.dialogue_column].replace(np.nan, '', regex=True)

    def encodeColumnUsingDictInListOfDataFrames(self, list_of_dfs, column_name, dictionary: dict):
        """
        For each DataFrame in a list of DataFrames, encodes a column's entries using a dictionary
        :param list_of_dfs: A list of DataFrames that includes a column_name column
        :param column_name: The name of the column to encode
        :param dictionary:  The dictionary were the key is the item to encode and the value is the new desired encoded value
        """
        for index in range(len(list_of_dfs)):
            list_of_dfs[index][column_name] = list_of_dfs[index][column_name].map(dictionary)

    def encodeColumnUsingDictInDataFrame(self, df, column_name: str, dictionary: dict):
        """
        In a DataFrame, encodes a column's entries using a dictionary
        :param df: A DataFrames that includes a column_name column
        :param column_name: The name of the column to encode
        :param dictionary:  The dictionary were the key is the item to encode and the value is the new desired encoded value
        """
        df[column_name] = df[column_name].map(dictionary)

    def dropNAFromListOfDataFrames(self, list_of_dfs, list_of_columns):
        """
        From a list of DataFrames, removes all rows of a DataFrame that contain NaN entries for any of the columns in
        the list_of_columns
        :param list_of_dfs: List of DataFrames to process
        :param list_of_columns: list of columns to check for NaN entries in
        """
        # Clean out NaN entries. We don't need this if we have perfect data.
        for index in range(len(list_of_dfs)):
            df = list_of_dfs[index]
            num_rows = df.shape[0]
            df[list_of_columns] = df[list_of_columns].replace('', np.nan, inplace=False)
            df.dropna(how='any', inplace=True, subset=list_of_columns)
            df.reset_index(drop=True, inplace=True)

            # Check how many entries were deleted
            print("Study " + str(index) + " - NA Rows: " + str(num_rows - df.shape[0]))

    def dropNAFromDataFrame(self, df: pd.DataFrame, list_of_columns: list):
        """
        Removes all rows of a DataFrame that contain NaN entries for any of the columns in
        the list_of_columns
        :param df: DataFrame to process
        :param list_of_columns: list of columns to check for NaN entries in
        """
        # Clean out NaN entries. We don't need this if we have perfect data.
        num_rows = df.shape[0]
        df[list_of_columns] = df[list_of_columns].replace('', np.nan, inplace=False)
        df.dropna(how='any', inplace=True, subset=list_of_columns)
        df.reset_index(drop=True, inplace=True)

        # Check how many entries were deleted
        print("NA Rows: " + str(num_rows - df.shape[0]))

    def deepCopyColumnsInListOfDataFrames(self, list_of_dfs: list, columns: list):
        """
        For every DataFrame in a list of DataFrames, creates a new DataFrames that is a deepcopy of all the 'columns'
        and adds it to a new list of DataFrames
        :param list_of_dfs: List of DataFrames to copy from
        :param columns: Which columns to copy over into the new DataFrames
        :return: A new list of DataFrames with deepcopied DataFrames
        """
        new_list_of_dfs = list()
        for df in list_of_dfs:
            new_df = df[columns]
            new_list_of_dfs.append(new_df.copy())
        return new_list_of_dfs

    def deepCopyColumnsInDataFrame(self, df: pd.DataFrame, columns: list):
        """
        Creates a new DataFrame that is a deepcopy of all the 'columns' of the original DataFrame
        :param df: DataFrame to copy from
        :param columns: Which columns to copy over into the new DataFrame
        :return: The deepcopied DataFrame
        """
        return df[columns].copy()

    def removeColumnFromListOfDataFrames(self, list_of_dfs: list, column_name: str):
        """
        Drops a single column in a list of DataFrames
        :param list_of_dfs: List of DataFrames
        :param column_name: The column to drop
        """
        for index in range(len(list_of_dfs)):
            list_of_dfs[index].drop(column_name, axis=1, inplace=True)

    def removeColumnFromDataFrame(self, df: pd.DataFrame, column_name: str):
        """
        Drops a single column of a DataFrame
        :param df: DataFrame
        :param column_name: The column to drop
        """
        df.drop(column_name, axis=1, inplace=True)

    def scaleColumninListOfDataFrames(self, list_of_dfs: list, column_name):
        """
        For each DataFrame in a list of DataFrames, scales all the values of a numerical column between 0 and 1
        :param list_of_dfs: List of DataFrames
        :param column_name: column to scale
        """
        for index in range(len(list_of_dfs)):
            x = list_of_dfs[index][column_name].values.reshape(-1, 1)
            scaler = preprocessing.MinMaxScaler()
            list_of_dfs[index][column_name] = scaler.fit_transform(x)

    def addPreviousDialogueToListOfDataFrames(self, list_of_dfs: list):
        """
        For each DataFrames in a list of DataFrames, adds the previous dialogue of the current speaker and the other
        person to each row (depending on the settings). If there is no previous dialogue, it is left blank.
        :param list_of_dfs: List of DataFrames
        """

        # Loop through all the studies
        for index in range(len(list_of_dfs)):
            # Create a copy of the original to edit
            if self.settings.same_prev:
                list_of_dfs[index][self.settings.same_prev_column] = ""
            if self.settings.other_prev:
                list_of_dfs[index][self.settings.other_prev_column] = ""

            # Loop through all the entries of the study
            for i in range(len(list_of_dfs[index])):
                # print("Current[" + str(i) + "]: " + context_df.loc[i, 'Dialogue Act'])

                # the current speaker
                speaker = list_of_dfs[index].loc[i, self.settings.speaker_column]
                # Use mapping scheme to vectorize
                # j is used to index previous rows to find the 2 previous dialogues of each participant
                j = i - 1

                # If we want to add a specific previous dialogue (same_prev = True),
                # we say that if is not yet found (same_found = False)
                same_found = not self.settings.same_prev
                other_found = not self.settings.other_prev

                # Loop through the past dialogues going backwards
                # Keep looking as long as we haven't found everything yet and we haven't reached the beginning of the study
                while (not same_found or not other_found) and j >= 0:
                    # the speaker for this past dialogue
                    compare_speaker = list_of_dfs[index].loc[j, self.settings.speaker_column]

                    # if this dialogue is the first of the same speaker
                    if same_found is False and compare_speaker == speaker:
                        same_found = True

                        # The jth row contains the ith row's previous dialogue from the same speaker
                        list_of_dfs[index].loc[i, self.settings.same_prev_column] = list_of_dfs[index].loc[j, self.settings.dialogue_column]

                        # print("Same[" + str(i) + "] = Original[" + str(j) + "]: " + context_df.loc[i, 'Same Previous Dialogue'])

                    # if this dialogue is the first of the other speaker
                    elif other_found is False and compare_speaker != speaker:
                        other_found = True
                        # The jth row contains the ith row's previous dialogue from the other speaker
                        list_of_dfs[index].loc[i, self.settings.other_prev_column] = list_of_dfs[index].loc[j, self.settings.dialogue_column]
                        # print("Other[" + str(i) + "] = Original[" + str(j) + "]: " + context_df.loc[i, 'Other P Previous Dialogue'])
                    j -= 1

    def addPreviousDialogueToDataFrame(self, df: pd.DataFrame):
        """
         adds the previous dialogue of the current speaker and the other person to each row of a DataFrame
         (depending on the settings). If there is no previous dialogue, it is left blank.
         :param list_of_dfs: List of DataFrames
         """
        # Create a copy of the original to edit
        if self.settings.same_prev:
            df[self.settings.same_prev_column] = ""
        if self.settings.other_prev:
            df[self.settings.other_prev_column] = ""

        # Loop through all the entries of the study
        for i in range(len(df)):
            # print("Current[" + str(i) + "]: " + context_df.loc[i, 'Dialogue Act'])

            # the current speaker
            speaker = df.loc[i, self.settings.speaker_column]
            # Use mapping scheme to vectorize
            # j is used to index previous rows to find the 2 previous dialogues of each participant
            j = i - 1

            # If we want to add a specific previous dialogue (same_prev = True),
            # we say that if is not yet found (same_found = False)
            same_found = not self.settings.same_prev
            other_found = not self.settings.other_prev

            # Loop through the past dialogues going backwards
            # Keep looking as long as we haven't found everything yet and we haven't reached the beginning of the study
            while (not same_found or not other_found) and j >= 0:
                # the speaker for this past dialogue
                compare_speaker = df.loc[j, self.settings.speaker_column]

                # if this dialogue is the first of the same speaker
                if same_found is False and compare_speaker == speaker:
                    same_found = True

                    # The jth row contains the ith row's previous dialogue from the same speaker
                    df.loc[i, self.settings.same_prev_column] = df.loc[
                        j, self.settings.dialogue_column]

                    # print("Same[" + str(i) + "] = Original[" + str(j) + "]: " + context_df.loc[i, 'Same Previous Dialogue'])

                # if this dialogue is the first of the other speaker
                elif other_found is False and compare_speaker != speaker:
                    other_found = True
                    # The jth row contains the ith row's previous dialogue from the other speaker
                    df.loc[i, self.settings.other_prev_column] = df.loc[
                        j, self.settings.dialogue_column]
                    # print("Other[" + str(i) + "] = Original[" + str(j) + "]: " + context_df.loc[i, 'Other P Previous Dialogue'])
                j -= 1

    def addOtherParticipantsInfoToListOfDataFrames(self, list_of_dfs: list, info_column: str, other_column: str):
        """
        For each DataFrame in a list of DataFrames, Adds a property of the other participant to each row as a new column
        We only use this for gender, but the function is generalized anyways
        :param list_of_dfs: List of DataFrames
        :param info_column: The column we want to include about the other participant
        :param other_column: name we want to call this new column
        """
        # Loop through all the studies
        for index in range(len(list_of_dfs)):
            list_of_dfs[index][other_column] = ""
            # Loop through all the entries of the study
            for i in range(len(list_of_dfs[index])):
                speaker = list_of_dfs[index].loc[i, self.settings.speaker_column]
                gender_found = False
                j = 0
                # Start from the beginning and look for the other speaker's gender
                while not gender_found and j < len(list_of_dfs[index].index):
                    # the speaker for this past dialogue
                    compare_speaker = list_of_dfs[index].loc[j, self.settings.speaker_column]
                    # if this dialogue is of the other speaker
                    if compare_speaker != speaker:
                        gender_found = True
                        # Assign this gender as the other person's gender in the original row
                        list_of_dfs[index].loc[i, other_column] = list_of_dfs[index].loc[j, info_column]
                    j += 1

    def addOtherParticipantsInfoToDataFrame(self, df: pd.DataFrame, info_column: str, other_column: str):
        """
        Adds a property of the other participant to each row as a new column in a DataFrame
        We only use this for gender, but the function is generalized anyways
        :param df: DataFrame
        :param info_column: The column we want to include about the other participant
        :param other_column: name we want to call this new column
        """
        df[other_column] = ""
        # Loop through all the entries
        for i in range(len(df)):
            speaker = df.loc[i, self.settings.speaker_column]
            gender_found = False
            j = 0
            # Start from the beginning and look for the other speaker's gender
            while not gender_found and j < len(df.index):
                # the speaker for this past dialogue
                compare_speaker = df.loc[j, self.settings.speaker_column]
                # if this dialogue is of the other speaker
                if compare_speaker != speaker:
                    gender_found = True
                    # Assign this gender as the other person's gender in the original row
                    df.loc[i, other_column] = df.loc[j, info_column]
                j += 1

    def tfidfListOfDataFrames(self, list_of_dfs):
        """
        Apply tfidf to create a lot of new columns to each dialogue column we are using (specified in the settings)
        :param list_of_dfs: List of DataFrames
        """
        # Put all current dialogues into a single series for tfidf training
        all_dialogue_series = pd.Series()
        for features in list_of_dfs:
            if features[self.settings.dialogue_column].isnull().sum() > 0:
                print("Found NaN: " + str(features[self.settings.dialogue_column].isnull().sum()))
            all_dialogue_series = all_dialogue_series.append(features[self.settings.dialogue_column])

        # Create tf-idf vectorizer with properties specified in the settings
        tfidf_vectorizer = TfidfVectorizer(max_df=self.settings.max_freq, min_df=self.settings.min_freq, ngram_range=(1, self.settings.max_gram))

        # Fit only using the current dialogues to make document frequency normal
        tfidf_vectorizer.fit(all_dialogue_series)

        print("Number of tfidf features: " + str(len(tfidf_vectorizer.get_feature_names())))
        print(tfidf_vectorizer.get_feature_names())
        if self.settings.dialogue:
            self.tfidfColumnInListOfDataFrames(tfidf_vectorizer, list_of_dfs, self.settings.dialogue_column, '')
        if self.settings.same_prev:
            self.tfidfColumnInListOfDataFrames(tfidf_vectorizer, list_of_dfs, self.settings.same_prev_column, self.settings.same_prev_prepend)
        if self.settings.other_prev:
            self.tfidfColumnInListOfDataFrames(tfidf_vectorizer, list_of_dfs, self.settings.other_prev_column, self.settings.other_prev_prepend)

    def tfidfDataFrame(self, df: pd.DataFrame):
        """
        Apply tfidf to create a lot of new columns to each dialogue column we are using (specified in the settings)
        :param list_of_dfs: List of DataFrames
        """
        # Put all current dialogues into a single series for tfidf training
        all_dialogue_series = pd.Series()
        if df[self.settings.dialogue_column].isnull().sum() > 0:
            print("Found NaN: " + str(df[self.settings.dialogue_column].isnull().sum()))
        all_dialogue_series = all_dialogue_series.append(df[self.settings.dialogue_column])
        print(all_dialogue_series.shape)

        # Create tf-idf vectorizer with properties specified in the settings
        tfidf_vectorizer = TfidfVectorizer(max_df=self.settings.max_freq, min_df=self.settings.min_freq, ngram_range=(1, self.settings.max_gram))

        # Fit only using the current dialogues to make document frequency normal
        tfidf_vectorizer.fit(all_dialogue_series)

        print("Number of tfidf features: " + str(len(tfidf_vectorizer.get_feature_names())))
        print(tfidf_vectorizer.get_feature_names())
        if self.settings.dialogue:
            df = self.tfidfColumnInDataFrame(tfidf_vectorizer, df, self.settings.dialogue_column, '')
        if self.settings.same_prev:
            df = self.tfidfColumnInDataFrame(tfidf_vectorizer, df, self.settings.same_prev_column, self.settings.other_prev_prepend)
        if self.settings.other_prev:
            df = self.tfidfColumnInDataFrame(tfidf_vectorizer, df, self.settings.other_prev_column, self.settings.other_prev_prepend)
        return df

    def tfidfColumnInListOfDataFrames(self, tfidf_vectorizer: TfidfVectorizer, list_of_dfs: List[pd.DataFrame], column_name, prepend_to_label: str):
        """
        For each DataFrame in a list of DataFrames, apply a tfidf_vectorizer to a particular column
        :param tfidf_vectorizer TfidfVectorizer
        :param list_of_dfs: List of DataFrames
        :param column_name: name of dialogue column to apply tfidf
        :param prepend_to_label: The string to prepend to all column titles created from the tfidf
        """
        tfidf_matrix_list = list()

        for i in range(len(list_of_dfs)):
            # Transform all three to different matrices
            tfidf_matrix_list.append(tfidf_vectorizer.transform(list_of_dfs[i][column_name]).toarray())
        tfidf_labels = tfidf_vectorizer.get_feature_names()

        if not column_name == "":
            tfidf_labels = list(map(lambda x: prepend_to_label + x, tfidf_labels))

        for i in range(len(list_of_dfs)):
            list_of_dfs[i].drop(column_name, axis=1, inplace=True)
            list_of_dfs[i] = list_of_dfs[i].join(pd.DataFrame(data=tfidf_matrix_list[i], columns=tfidf_labels))

    def tfidfColumnInDataFrame(self, tfidf_vectorizer, df: pd.DataFrame, column_name, prepend_to_label: str):
        """
        apply a tfidf_vectorizer to a particular column in a DataFrame
        :param tfidf_vectorizer TfidfVectorizer
        :param df: DataFrame
        :param column_name: name of dialogue column to apply tfidf
        :param prepend_to_label: The string to prepend to all column titles created from the tfidf
        """
        # Transform to matrix
        tfidf_matrix = tfidf_vectorizer.transform(df[column_name]).toarray()
        tfidf_labels = tfidf_vectorizer.get_feature_names()

        if not column_name == "":
            tfidf_labels = list(map(lambda x: prepend_to_label + x, tfidf_labels))

        df.drop(column_name, axis=1, inplace=True)
        df = df.join(pd.DataFrame(data=tfidf_matrix, columns=tfidf_labels))
        return df

    def preprocessCategoricalWordInListOfDataFrames(self, list_of_dfs: list, column_name):
        """
        A general preprocessing function for categorical data for each DataFrame in a list of DataFrames
        Removes all spaces, lowercase, remove punctuation
        :param list_of_dfs: List of DataFrames
        :param column_name: name of the column to process
        """
        for i in range(len(list_of_dfs)):
            list_of_dfs[i][column_name] = list_of_dfs[i][column_name].apply(self.preprocessCategoricalWord)

    def preprocessCategoricalWordInDataFrame(self, df: pd.DataFrame, column_name):
        """
        A general preprocessing function for categorical data for a DataFrame
        Removes all spaces, lowercase, remove punctuation
        :param df: DataFrame
        :param column_name: name of the column to process
        """
        df[column_name] = df[column_name].apply(self.preprocessCategoricalWord)

    def encodeColumnInListOfDataFrames(self, list_of_dfs, column_name):
        """
        Encode a categorical column into numbers for each DataFrame in a list of DataFrames
        :param list_of_dfs: List of DataFrames
        :param column_name: the name of the column to encode
        """
        # accumulate a list of all tags for label encoding
        label_data = pd.Series()

        # Preprocess dialogue tags
        for i in range(len(list_of_dfs)):
            label_data = label_data.append(list_of_dfs[i][column_name])

        # Use LabelEncoder to vectorize the 'Tags'
        encoder = LabelEncoder()
        encoder.fit(label_data)

        for i in range(len(list_of_dfs)):
            list_of_dfs[i][column_name] = encoder.transform(list_of_dfs[i][column_name])

        self.settings.label_dict = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

    def encodeColumnInDataFrame(self, df, column_name):
        """
        Encode a categorical column into numbers for a DataFrame
        :param df: DataFrame
        :param column_name: the name of the column to encode
        """
        # Use LabelEncoder to vectorize the 'Tags'
        encoder = LabelEncoder()
        encoder.fit(df[column_name])

        df[column_name] = encoder.transform(df[column_name])

        self.settings.label_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    def isDriver(self, speaker, driver):
        """
        Depending on the speaker and driver column, determine if one speaking is the driver
        :param speaker: value for the speaker Ex: 0, 1 or P1, P2
        :param driver: value for the driver Ex: 0, 1 or P1, P2
        :return: new driver value
        """
        if speaker == driver:
            return 0  # Speaker is the driver
        elif driver == 'Collaborating' or driver == 'C':
            return 1  # Speaker is collaborating
        else:
            return 2  # Speaker is the navigator

    def addDriverToListOfDataFrames(self, list_of_dfs: list):
        """
        add a new column that says whether the current speaker is the driver to each DataFrame in a list of DataFrames
        :param list_of_dfs: List of DataFrames
        """
        for i in range(len(list_of_dfs)):
            list_of_dfs[i][self.settings.driver_column] = list_of_dfs[i].apply(lambda x: self.isDriver(x[self.settings.speaker_column], x[self.settings.driver_column]), axis=1)
        self.dropNAFromListOfDataFrames(list_of_dfs, [self.settings.driver_column])
        print(str(list_of_dfs[0].columns))

    def addDriverToDataFrame(self, df: pd.DataFrame):
        """
        add a new column that says whether the current speaker is the driver to the DataFrame
        :param list_of_dfs: List of DataFrames
        """
        df[self.settings.driver_column] = df.apply(lambda x: self.isDriver(x[self.settings.speaker_column], x[self.settings.driver_column]), axis=1)
        self.dropNAFromDataFrame(df, [self.settings.driver_column])
        print(str(df.columns))

    def ListToDF(self, list_of_dfs: list):
        """
        Merge a list of DataFrames into a single DataFrame
        :param list_of_dfs: List of DataFrames
        :return: DataFrame a single DataFrame that contains the same information as all inputted DataFrames
        """
        result = pd.DataFrame()
        for df in list_of_dfs:
            result = result.append(df)
        return result

    def nFoldHMM(self, list_of_dfs):
        """
        Does n-fold cross validation with HMM, prints the accuracies, and displays a confusion matrix
        :param list_of_dfs: List of DataFrames
        """
        train_features_list = list()
        test_features = pd.DataFrame()
        num_folds = len(list_of_dfs)
        accuracies = list()
        prediction_list = list()
        true_values = list()
        for i in range(num_folds):

            # Copy the features_list so that the list is in tact next loop
            fold_features_list = list_of_dfs.copy()

            # The test is removed from this fold's list
            # The result is NOT a list, it is a DataFrame
            test_features = fold_features_list.pop(i).copy()

            # The train list is the original list after the testing data is removed
            train_features_list = fold_features_list

            train_labels_list = list()
            train_study_lengths = list()
            for j in range(len(train_features_list)):
                # Create a copy of the j'th Dataframe
                train_features = train_features_list[j].copy()

                # print("train_features.columns: " + str(train_features.columns))
                # Copy the label out of both the testing a training DataFrames
                train_labels_list.append(train_features[self.settings.label_column])

                # Remove the original column from the feature DataFrame
                train_features.drop(self.settings.label_column, axis=1, inplace=True)
                train_features_list[j] = train_features

                # Add the length of the study to a list
                train_study_lengths.append(len(train_features.index))

            test_labels = pd.DataFrame()

            test_length = [len(test_features.index)]

            test_labels = test_features[self.settings.label_column]
            test_features.drop(self.settings.label_column, axis=1, inplace=True)

            train_features = pd.DataFrame()
            train_labels = pd.Series()

            for k in range(len(train_features_list)):
                train_features = train_features.append(train_features_list[k], ignore_index=True)
                train_labels = train_labels.append(train_labels_list[k], ignore_index=True)

            hmm = MultinomialHMM(alpha=0.01)
            hmm.fit(train_features, train_labels, train_study_lengths)
            prediction = hmm.predict(test_features, test_length)
            prediction_list.extend(prediction)
            accuracy = accuracy_score(test_labels.values, prediction)
            true_values.extend(test_labels.values)
            #for prediction, real in zip(prediction, test_labels.values):
            #    print(str(prediction) + ", " + str(real))
            print("Fold " + str(i) + " - Accuracy: {:0.2f}".format(accuracy * 100))
            accuracies.append(accuracy)
        total_accuracy = 0
        for num in accuracies:
            total_accuracy += num
        total_accuracy /= len(accuracies)
        print("Total Accuracy: {:0.2f}".format(total_accuracy * 100))
        self.confusionMatrix(true_values, prediction_list)


    def SVM(self, df: pd.DataFrame):
        """
        Uses a simple train-test split and does SVM. Prints accuracy and a confusion matrix
        :param list_of_dfs: List of DataFrames
        """
        labels = df.loc[:, self.settings.label_column]
        features = df.loc[:, df.columns != self.settings.label_column]

        # Split all the data into train and test
        feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=0)

        print("Train: " + str(feature_train.shape))
        print("Test: " + str(feature_test.shape))

        #print(feature_test['Timestamp Start'])
        print("Running SVM")
        start_time = time.time()
        print(feature_train)
        print(label_train)
        # Run a basic linear SVM.
        clf = None

        if self.settings.class_weights:
            clf = svm.SVC(kernel=self.settings.kernel, degree=self.settings.poly_degree, class_weight='balanced')
        else:
            clf = svm.SVC(kernel=self.settings.kernel, degree=self.settings.poly_degree)

        clf.fit(feature_train, label_train)
        elapsed_time = time.time() - start_time
        print("Time: " + str(elapsed_time))
        # Test predictions and calculate accuracy
        clf_predictions = clf.predict(feature_test)
        print("Accuracy: {:0.2f}%".format(clf.score(feature_test, label_test) * 100))
        self.confusionMatrix(label_test, clf_predictions)

    def tally(self, df: pd.DataFrame):
        """
        Adds up the number of occurrences of the values of the label and prints the result
        :param df: DataFrame
        """
        count = Counter(df[self.settings.label_column])
        tally = dict((self.settings.label_dict[key], value) for (key, value) in count.items())
        print("Tally: " + str(tally))
        total_items = sum(tally.values())
        tally_percent = dict((self.settings.label_dict[key], "{:0.2f}%".format((value / total_items) * 100)) for (key, value) in count.items())
        print("Tally Percent: " + str(tally_percent))


    def confusionMatrix(self, true_values, predicted_values):
        """
        Displays a confusion matrix given predictions and ground truth
        :param true_values: The ground truth. The real values
        :param predicted_values: The values that were predicted from the algorithm
        """
        sorted_label_index_list = sorted(self.settings.label_dict, key=self.settings.label_dict.get)
        sorted_label_list = sorted(self.settings.label_dict.values())
        print(sorted_label_index_list)
        print(sorted_label_list)
        cm = confusion_matrix(true_values, predicted_values, labels=sorted_label_index_list, sample_weight=None)
        if self.settings.normalized_heatmap:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(cm, sorted_label_list,  sorted_label_list)
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        heatmap = None
        if self.settings.normalized_heatmap:
            heatmap = sn.heatmap(df_cm, annot=False, fmt='f', annot_kws={"size": 16}, xticklabels=sorted_label_list,
                       yticklabels=sorted_label_list)  # font size
        else:
            heatmap = sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 16}, xticklabels=sorted_label_list,
                       yticklabels=sorted_label_list)  # font size
        heatmap.set(xlabel='predicted', ylabel='actual')
        plt.show()

    def run(self, original_list: list):
        self.dropNAFromListOfDataFrames(original_list, self.settings.column_list)

        new_list = self.deepCopyColumnsInListOfDataFrames(original_list, self.settings.column_list)

        # Scale all the time values between 0 and 1
        if self.settings.time_start:
            self.scaleColumninListOfDataFrames(new_list, self.settings.time_start_column)
        if self.settings.time_end:
            self.scaleColumninListOfDataFrames(new_list, self.settings.time_end_column)
        if self.settings.time_elapsed:
            self.scaleColumninListOfDataFrames(new_list, self.settings.time_elapsed_column)
        if self.settings.time_between:
            self.scaleColumninListOfDataFrames(new_list, self.settings.time_between_column)

        # Add the driver column if selected
        if self.settings.driver:
            self.addDriverToListOfDataFrames(new_list)
            #print("addDriverToListOfDataFrames")

        # Encode the gender column if either the current gender or the other's gender is selected
        if self.settings.gender or self.settings.other_gender:
            self.encodeColumnUsingDictInListOfDataFrames(new_list, self.settings.gender_column, self.settings.gender_dict)
            #print("preprocessGenderInListOfDataFrames")

        # Add the other gender's info if selected
        if self.settings.other_gender:
            self.addOtherParticipantsInfoToListOfDataFrames(new_list, self.settings.gender_column, self.settings.other_gender_column)
            #print("addOtherGenderToListOfDataFrames")

            # Remove the gender column if it is not selected (We only needed it temporarily)
            if not self.settings.gender:
                self.removeColumnFromListOfDataFrames(new_list, self.settings.gender_column)

        # If just one dialogue option is selected
        if self.settings.dialogue or self.settings.other_prev or self.settings.same_prev:

            # Preprocess the current dialogue
            #print("Preprocesssing Dialogue")
            self.preprocessDialogueInListOfDataFrames(new_list)

            # If previous dialogue settings are enabled
            if self.settings.other_prev or self.settings.same_prev:

                # Add previous dialogue
                self.addPreviousDialogueToListOfDataFrames(new_list)

            # tfidf the dialogue based on the settings
            self.tfidfListOfDataFrames(new_list)

            # If the current dialogue settings is disabled
            if not self.settings.dialogue:

                # Remove the current dialogue
                self.removeColumnFromListOfDataFrames(new_list, self.settings.dialogue_column)

        # If the driver is the label, then we have to use a special function instead of the generic one
        if self.settings.label_column == self.settings.driver_column:
            self.addDriverToListOfDataFrames(new_list)
        else:
            # Preprocess the label column with a generic categorical preprocessor (lowercase, remove spaces and punctuation)
            self.preprocessCategoricalWordInListOfDataFrames(new_list, self.settings.label_column)

        # Remove the speaker column if it was added previously
        if self.settings.need_speaker:
            self.removeColumnFromListOfDataFrames(new_list, self.settings.speaker_column)

        # Encode the label column using a categorical encoder
        self.encodeColumnInListOfDataFrames(new_list, self.settings.label_column)

        # Merge all dataframes into one (used for tallying and SVM)
        df = self.ListToDF(new_list)
        #TODO print("Time: " + str(df.loc[0, self.settings.time_elapsed_column]))

        # Tally the values in the label column
        self.tally(df)

        # Either run the SVM of HMM functions depending on the settings
        if self.settings.svm:
            self.SVM(df)
        else:
            self.nFoldHMM(new_list)