import pandas as pd
import os


class OpalDataSet:
    def __init__(self):
        """
        self.data_list: overall data split in to 8 chunks, the first 7 chunks are used for training, the other for testing
        every chunk is  a tuple: (temp_old, temp_real, temp_fake)
        temp_old shape: (num_passengers * num_historical_records, x_dim)
        temp_real shape: (num_passengers, x_dim)
        temp_fake shape: (num_passemgers, x_dim)
        """
        old_path = '/share/scratch/xuesongwang/opal_data/data_pair/old'
        real_path = '/share/scratch/xuesongwang/opal_data/data_pair/real'
        fake_path = '/share/scratch/xuesongwang/opal_data/data_pair/fake'

        # for testing different features
        self.numer_list = ['JS_DURN_SEC','JS_FARE_CENTS_AMT','TAG2_DIST_BAND_CD',
                      'TAG1_DT_FK','TAG1_TM','TAG1_LAT_VAL','TAG1_LONG_VAL','TAG1_TRNFR_IND',
                      'TAG2_DT_FK','TAG2_TM','TAG2_LAT_VAL','TAG2_LONG_VAL','TAG2_TRNFR_IND']

        self.categ_list = ['TS_TYP_CD','IMTT_CD','ROUTE_ID','ROUTE_VAR_ID','RUN_DIR_CD',
                           'TAG1_TS_PC','TAG1_TS_NUM','TAG2_TS_PC','TAG2_TS_NUM']

        files = os.listdir(old_path)
        data_list = []
        for i, file in enumerate(files):
            print (file)
            temp_old = pd.read_csv(old_path + "/" + file)
            temp_real = pd.read_csv(real_path + "/" + file)
            temp_fake = pd.read_csv(fake_path + "/" + file)
            data_list.append((temp_old, temp_real, temp_fake))
        self.train = data_list[:-1]
        self.test = data_list[-1]
        self.overall = data_list
        print('finish concate!')

    def __len__(self):
        return len(self.train)

    def getdata(self, chunk):
        # normally only training data need to be sclied
        return self.train[0][chunk], self.train[1][chunk], self.train[2][chunk]



if __name__ == '__main__':
    dataloader = OpalDataSet()
    print ("done!")

