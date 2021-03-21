﻿import arrow
import json
import base64
import requests
import logging
import os
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import time
import pickle
import pandas as pd

maxdatapoints = 100000  # Datapoints per request, limited by Myplant


def epoch_ts(ts) -> float:
    if ts >= 10000000000.0:
        return float(ts/1000.0)
    else:
        return float(ts)


def mp_ts(ts) -> int:
    if ts >= 10000000000.0:
        return int(ts)
    else:
        return int(ts * 1000.0)


class MyPlantException(Exception):
    pass


burl = 'https://api.myplant.io'
errortext = {
    200: 'successful operation',
    400: 'Request is missing required HTTP header \'x-seshat-token\'',
    401: 'The supplied authentication is invalid',
    403: 'No permission to access this resource',
    404: 'No data was found',
    500: 'Internal Server Error'
}


class MyPlant(object):

    _name = ''
    _password = ''
    _session = None
    _caching = 0

    def __init__(self, caching=7200):
        """MyPlant Constructor"""
        self._caching = caching
        # load and manage credentials from hidden file
        try:
            with open("./data/.credentials", "r", encoding='utf-8-sig') as file:
                cred = json.load(file)
            self._name = cred['name']
            self._password = cred['password']
        except FileNotFoundError:
            raise

    @ classmethod
    def load_dataitems_csv(cls, filename):
        """load CSV dataitems definition file

        example content:
        ID;myPlantName;unit
        52;Exhaust_TempCylAvg;C (high)
        103;Various_Values_PosTurboBypass;%
        19079;Exhaust_TCSpeedA;1000/min
        ....

        Args:
            filename (string): CSV dataitems definition file

        Returns:
            dict: CSV dataitems dict
        """
        data_req = pd.read_csv("DataItems_Request.csv",
                               sep=';', encoding='utf-8')
        dat = {a[0]: [a[1], a[2]] for a in data_req.values}
        return dat

    def deBase64(self, text):
        return base64.b64decode(text).decode('utf-8')

    def gdi(self, ds, sub_key, data_item_name):
        """Unpack value from Myplant Json datastructure based on key & DataItemName"""
        if sub_key == 'nokey':
            return ds.get(data_item_name, None)
        else:
            local = {x['value']
                     for x in ds[sub_key] if x['name'] == data_item_name}
            return local.pop() if len(local) != 0 else None

    @property
    def caching(self):
        """the current cache time"""
        return self._caching

    def login(self):
        """Login to MyPlant"""
        if self._session is None:
            logging.debug(f"SSO {self.deBase64(self._name)} MyPlant login")
            self._session = requests.session()
            headers = {'Content-Type': 'application/json', }
            body = {
                "username": self.deBase64(self._name),
                "password": self.deBase64(self._password)
            }
            loop = 1
            try:
                while loop < 3:
                    response = self._session.post(burl + "/auth",
                                                  data=json.dumps(body), headers=headers)
                    if response.status_code == 200:
                        logging.debug(f'login {self._name} successful.')
                        break
                    else:
                        logging.error(
                            f'login failed with response code {response.status_code}')
                    loop += 1
                    logging.error(f'Myplant login attempt #{loop}')
                    time.sleep(1)
                if loop >= 3:
                    logging.error(f'Login {self._name} failed')
                    raise MyPlantException(
                        f'Login {self._name} failed')
            except:
                raise

    def logout(self):
        """Logout from Myplant and release self._session"""
        if self._session != None:
            self._session.close()
            self._session = None

    def fetchdata(self, url):
        """login and return data based on url"""
        try:
            self.login()
            logging.debug(f'url: {url}')
            response = self._session.get(burl + url)
            if response.status_code == 200:
                logging.debug(f'fetchdata: download successful')
                res = response.json()
                return res
            else:
                logging.error(
                    f' Code: {url}, {response.status_code}, {errortext[response.status_code]}')
        except:
            raise

    def _asset_data(self, serialNumber):
        """
        Returns an Asset based on its id with all details
        including properties and DataItems.

        Parameters:
        Name	    type    Description
        sn          int     IB ItemNumber Engine
        ----------------------------------------------
        url: /asset?assetType=J-Engine&serialNumber=sn
        """
        return self.fetchdata(url=r"/asset?assetType=J-Engine&serialNumber=" + str(serialNumber))

    def historical_dataItem(self, id, itemId, timestamp):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type    Description
        assetId     int64   Id of the Asset to query the DateItem for.
        dataItemId  int64   Id of the DataItem to query.
        timestamp   int64   Optional,  timestamp in the DataItem history to query for.
        highres     Boolean Whether to use high res data. Much slower but gives the raw data.
        """
        return self.fetchdata(url=fr"/asset/{id}/dataitem/{itemId}?timestamp={timestamp}")

    def history_dataItem(self, id, itemId, p_from, p_to, timeCycle=3600):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type    Description
        assetId     int64   Id of the Asset to query the DateItem for.
        dataItemId  int64   Id of the DataItem to query.
        p_from      int64   timestamp start timestamp.
        p_to        int64   timestamp stop timestamp.
        timeCycle   int64   interval in seconds.
        """
        return self.fetchdata(url=fr"/asset/{id}/history/data?from={p_from}&to={p_to}&assetType=J-Engine&dataItemId={itemId}&timeCycle={timeCycle}&includeMinMax=false&forceDownSampling=false")

    def _history_batchdata(self, id, itemIds, lp_from, lp_to, timeCycle=3600):
        # comma separated string of DataItemID's
        IDS = ','.join([str(s) for s in itemIds.keys()])
        ldata = self.fetchdata(
            url=fr"/asset/{id}/history/batchdata?from={lp_from}&to={lp_to}&timeCycle={timeCycle}&assetType=J-Engine&includeMinMax=false&forceDownSampling=false&dataItemIds={IDS}")
        # restructure data to dict
        ds = dict()
        ds['labels'] = ['time'] + [itemIds[x][0]
                                   for x in ldata['columns'][1]]
        ds['data'] = [[r[0]] + [rr[0] for rr in r[1]]
                      for r in ldata['data']]
        # import data to Pandas DataFrame and return result
        df = pd.DataFrame(ds['data'], columns=ds['labels'])
        return df

    def hist_data(self, id, itemIds, p_from, p_to, timeCycle=3600
                  # , cui_log=True
                  ):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type            Description
        assetId     int64           Id of the Asset to query the DateItem for.
        itemIds     dict            DataItem Id's, Names & Units
        p_from      int64           timestamp start timestamp in ms.
        p_to        int64           timestamp stop timestamp in ms.
        timeCycle   int64           interval in seconds.
        cui_log     boolean         report progress on CUI or not
        """

        # initialize data collector
        df = pd.DataFrame([])
        # calculate how many full rows per request within the myplant limit are possible
        rows_per_request = maxdatapoints // len(itemIds)
        rows_total = (p_to.timestamp -
                      p_from.timestamp) / timeCycle

        print(
            f"Rows per Request: {rows_per_request}, cycle per row: {timeCycle} s, total rows: {rows_total}, {(rows_total * 30) / 3600:0.2f} oph's")

        pbar = tqdm(total=rows_total)  # count in minutes

        # initialize loop
        lp_from = p_from.timestamp * 1000  # Start at p_from
        lp_to = min((lp_from + rows_per_request * timeCycle * 1000),
                    p_to.timestamp * 1000)

        while lp_from < p_to.timestamp * 1000:
            # if cui_log:
            #     print(
            #         f"chunk {arrow.get(lp_from).format('DD.MM.YYYY - HH:mm')} to {arrow.get(lp_to).format('DD.MM.YYYY - HH:mm')}")
            ldf = self._history_batchdata(
                id, itemIds, lp_from, lp_to, timeCycle)
            # and append each chunk to the return df
            df = df.append(ldf)
            pbar.update(rows_per_request)
            # calculate next cycle
            lp_from = lp_to + timeCycle * 1000
            lp_to = min((lp_to + rows_per_request *
                         timeCycle * 1000), p_to.timestamp * 1000)

        pbar.close()
        # Addtional Datetime column calculated from timestamp
        df['datetime'] = pd.to_datetime(df['time'] * 1000000)
        return df
