from Settings import Settings
from SheetToDataFrame import SheetToDataFrame
from Classifier import Classifier
import pandas as pd



class Runner:

    def __init__(self):
        self.settings = Settings('settings.json')
        self.classifier = Classifier(self.settings)
        self.sheet_getter = SheetToDataFrame(self.settings)

        pd.set_option('display.max_columns', 20)
        pd.set_option('display.max_rows', 10)

    def run(self):
        original_list = self.sheet_getter.get_data_as_list_of_dfs()
        self.classifier.run(original_list)


runee = Runner()
runee.run()





