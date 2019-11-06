from __future__ import print_function
import pickle
import os.path
from typing import List

import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from Settings import Settings


class SheetToDataFrame:
    """
    A class that queries a Google API and retrieves the Google Sheet:
    https://docs.google.com/spreadsheets/d/1eJzUfA5vkJ4zPDhvEkXnS2gOIGoEuZZoCCtF8CbywZ4/edit?usp=sharing
    Two functions exist to get the sheets in two different formats
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sheets = self.__get_service().spreadsheets()

        # The ID and range of a sample spreadsheet.
        self.SPREADSHEET_ID = '1eJzUfA5vkJ4zPDhvEkXnS2gOIGoEuZZoCCtF8CbywZ4'

    @staticmethod
    def get_sec(time_str):
        """
        A static method that converts hr:min:sec.### (str) to seconds (float)
        Must be static to be used in apply()
        :parameter time_str a string with the format hr:min:sec.###
        :returns float time in seconds
        """
        h, m, s = time_str.split(':')
        if '.' in s:
            s, ms = s.split('.')
            return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) * (1 / 100.0)
        else:
            return float(h) * 3600 + float(m) * 60 + float(s)

    def __adjust_timestamps(self, sheet):
        """
        Converts time string to float for all time columns
        Makes the 'Timestamp Start' and 'Timestamp End' columns start at 0:00:00
        :parameter DataFrame
        :returns DataFrame
        """

        sheet[self.settings.time_start_column] = sheet[self.settings.time_start_column].apply(SheetToDataFrame.get_sec)
        sheet[self.settings.time_end_column] = sheet[self.settings.time_end_column].apply(SheetToDataFrame.get_sec)
        sheet[self.settings.time_elapsed_column] = sheet[self.settings.time_elapsed_column].apply(SheetToDataFrame.get_sec)
        sheet[self.settings.time_between_column] = sheet[self.settings.time_between_column].apply(SheetToDataFrame.get_sec)

        # Subtract all values by the very first timestamp
        print("    Original Start Time: " + str(sheet[self.settings.time_start_column].iloc[0]))
        sheet[self.settings.time_start_column] = sheet[self.settings.time_end_column] - sheet[self.settings.time_start_column].iloc[0]
        sheet[self.settings.time_start_column] = sheet[self.settings.time_start_column] - sheet[self.settings.time_start_column].iloc[0]
        print("    New Start Time: " + str(sheet[self.settings.time_start_column].iloc[0]))

        return sheet

    def __get_service(self):
        """
        Get Google Sheets API credentials and get the service
        :returns service object
        """
        # If modifying these scopes, delete the file token.pickle.
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        creds = None

        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server()

            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('sheets', 'v4', credentials=creds)

    def __get_sheet(self, sheet_name: str) -> pd.DataFrame:
        """
        Retrieves sheet data from Google Sheets API
        :parameter sheets Google API service for Google Sheets
        :parameter SPREADSHEET_ID ID of the spreadsheet containing the data
        :parameter sheet_num the number of the study we want to retrieve
        :returns DataFrame of the rows and label
        """
        print("Querying " + sheet_name)

        # A query to get all the rows of the 'sheet_num'-th sheet
        ROWS_RANGE = sheet_name + '!A:K'

        try:
            # Apply query
            result = self.sheets.values().get(spreadsheetId=self.SPREADSHEET_ID, range=ROWS_RANGE).execute()

            print("    Result Found")

            # 2D List containing rows
            sheet_values = result.get('values', [])

            # Get the labels
            labels = sheet_values.pop(0)
            print(labels)

            if not sheet_values:
                print("    Empty")
                return pd.DataFrame()

            # Create DataFrame
            sheet = pd.DataFrame.from_records(sheet_values, columns=labels)

            # Preprocess TimeStamps
            return self.__adjust_timestamps(sheet)

        except HttpError:
            print("    HttpError: " + sheet_name + " not found")

            return pd.DataFrame()

    def get_data_as_list_of_dfs(self) -> List[pd.DataFrame]:
        """
        Creates a list of DataFrames where each DataFrame is an individual study
        :return: list of DataFrames
        """

        list_of_dfs = list()

        # Loop through Sheets and append each to original_df
        for sheet_name in self.settings.sheet_names:

            # Get the sheet as a DataFrame
            sheet = self.__get_sheet(sheet_name)

            # Check if the sheet is empty
            if sheet.empty:
                print("    Sheet is empty!")

            # If the sheet has data
            else:
                # Add the Data
                list_of_dfs.append(sheet)

                print("    sheet_df.shape: " + str(sheet.shape))
                print("    list_of_dfs.length: " + str(len(list_of_dfs)))

            print("")
        return list_of_dfs

    def get_data_as_df(self) -> pd.DataFrame:
        """
        All sheets are entered into a single DataFrame
        :return: DataFrame all studies combined
        """

        # Declare loop variables
        df = pd.DataFrame()

        # Loop through Sheets and append each to original_df
        for sheet_name in self.settings.study_names:

            # Get the sheet as a DataFrame
            sheet = self.__get_sheet(sheet_name)

            # If there is a sheet and it contains data
            if sheet.empty:
                print("    sheet is empty")

            else:
                # Add rows to original_df
                df = df.append(sheet)
                print("    sheet_df.shape: " + str(sheet.shape))
                print("    df.length: " + str(df.shape))

            print("")
        return df
