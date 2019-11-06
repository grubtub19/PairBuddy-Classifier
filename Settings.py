import json


class Settings:
    """
    Reads a json file of a specific format to get settings used throughout the program
    Makes it easy to try out different stuff
    """
    def __init__(self, filename: str):
        with open(filename) as f:
            json_file = json.load(f)

        # List of str: names of the individual sheets in the Google Sheet we want to use (Study6 is not done)
        self.sheet_names = json_file["sheet_names"]

        # All variables that end in '_column' are the column names Ex: 'Timestamp Start'
        self.time_start_column = json_file['time_start_column']
        self.time_end_column = json_file['time_end_column']
        self.time_elapsed_column = json_file['time_elapsed_column']
        self.time_between_column = json_file['time_between_column']
        self.gender_column = json_file['gender_column']
        self.other_gender_column = json_file['other_gender_column']
        self.speaker_column = json_file['speaker_column']
        self.driver_column = json_file['driver_column']
        self.dialogue_column = json_file['dialogue_column']
        self.other_prev_column = json_file['other_prev_column']
        self.same_prev_column = json_file['same_prev_column']

        # The str prepended to the tfidf column name. Ex: 'Other Prev: the dog'
        self.other_prev_prepend = json_file['other_prev_prepend']
        # The str prepended to the tfidf column name. Ex: 'Same Prev: the dog'
        self.same_prev_prepend = json_file['same_prev_prepend']

        # Name of the column we are predicting.
        # If it is already a column specified in another setting, make sure it's turned off there
        self.label_column = json_file['label_column']

        # Booleans that say whether or not the column is included as a feature
        self.time_start = json_file['time_start']
        self.time_end = json_file['time_end']
        self.time_elapsed = json_file['time_elapsed']
        self.time_between = json_file['time_between']
        self.gender = json_file['gender']
        self.other_gender = json_file['other_gender']
        self.dialogue = json_file['dialogue']
        self.other_prev = json_file['other_prev']
        self.same_prev = json_file['same_prev']
        self.driver = json_file['driver']

        # The dictionary that maps gender to numbers
        self.gender_dict = json_file['gender_dict']

        # Tfidf settings
        self.max_gram = json_file['max_gram']
        self.min_freq = json_file['min_freq']
        self.max_freq = json_file['max_freq']

        # True -> use SVM, False -> use HMM
        self.svm = json_file['svm']

        # These only apply when svm is being used
        self.kernel = json_file['kernel']
        self.poly_degree = json_file['poly_degree']

        # Do we want to use class weights
        self.class_weights = json_file['class_weights']

        # True -> confusion matrix uses percentages, False -> confusion matrix uses count
        self.normalized_heatmap = json_file['normalized_heatmap']

        # A list of the column names specified in the settings
        self.column_list = list()
        self.label_dict = dict()

        # Add time column names
        if self.time_start:
            self.column_list.append(self.time_start_column)
        if self.time_end:
            self.column_list.append(self.time_end_column)
        if self.time_elapsed:
            self.column_list.append(self.time_elapsed_column)
        if self.time_between:
            self.column_list.append(self.time_between_column)

        # A boolean to keep track if any of the settings require the current speaker to calculate
        # Ex: To determine the other speaker's previous dialogue, we need the speaker column temporarily
        self.need_speaker = False

        # If we are using any dialogue columns (even if it's a previous dialogue), we need the original dialogue to
        # calulate it.
        if self.dialogue or self.other_prev or self.same_prev:
            self.column_list.append(self.dialogue_column)

            # We require the speaker column if we need to determine the previous dialogues
            if self.other_prev or self.same_prev:
                self.need_speaker = True

        # If we want any gender info, we need the gender column
        if self.gender or self.other_gender:
            self.column_list.append(self.gender_column)

            # We only need the speaker when we need to determine the other speaker's gender
            if self.other_gender:
                self.need_speaker = True

        # The driver column says if the current speaker is the driver, so we need both the driver column and the speaker
        if self.driver:
            self.column_list.append(self.driver_column)
            self.need_speaker = True

        # We also need the speaker if our label is the driver (for the same reasons as before)
        if self.label_column == self.driver_column:
            self.need_speaker = True

        # Add the speaker column if we determined that we need it
        if self.need_speaker:
            print("Need Speaker")
            self.column_list.append(self.speaker_column)

        # Add the label column to the column list as well
        self.column_list.append(self.label_column)
