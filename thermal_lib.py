from scipy import signal, fft
from statsmodels.stats.multitest import fdrcorrection
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import os

hour2radians = lambda x: ((x%24)/24)*2*np.pi
radians2hour = lambda x: ((x%(2*np.pi))/(2*np.pi))*24
wrap2pi = lambda phases: (phases + np.pi) % (2 * np.pi) - np.pi


class Subjects():

    def __init__(self, pth=None):
        if pth is None:
            pth = Path(os.getcwd())/'thermal_data'

        if type(pth)==str:
            pth = Path(pth)

        self.pth = pth

        content = os.listdir(self.pth)
        content = list(filter(lambda x: (self.pth/x).is_dir(), content ) )
        content = list(filter(lambda x: x not in ['data_day','data_minute'], content ) )

        self.subjects = dict()
        for c in content:
            parts = c.split('-')
            sub_id = parts[2]
            if sub_id not in self.subjects.keys():
                self.subjects[sub_id] = dict()
                self.subjects[sub_id]['id'] = parts[2]
                subject_parts = self.subjects[sub_id]['id'].split('_')
                self.subjects[sub_id]['geno'] = subject_parts[0]
                self.subjects[sub_id]['number'] = subject_parts[1]
                self.subjects[sub_id]['rec_path'] = [(self.pth/c/'data.csv').as_posix()]
            else:
                self.subjects[sub_id]['rec_path'].append( (self.pth/c/'data.csv').as_posix() )

        self.subjects = pd.DataFrame.from_dict(self.subjects).T.reset_index(drop=True)
        self.subjects['rec_path'] = self.subjects['rec_path'].apply(sorted)
        self.subjects['rec_n'] = self.subjects['rec_path'].apply(len)

    @staticmethod
    def euclid(df):
        return np.sqrt((df['centroid_x'] - df['centroid_x'].shift(1))**2 +
                       (df['centroid_y'] - df['centroid_y'].shift(1))**2
                       )

    def __str__(self):
        string = 'Number of Subjects: ' + str( len(self.subjects) ) + '\n\n'
        for i,row in self.subjects.iterrows():
            string = string + str(i) + ' -> ID: ' + row['id'] + ' RECs: ' + str( row['rec_n'] ) + '\n'
        return string

    def __getitem__(self,idx):
        return self.subjects.loc[idx]

    def __len__(self):
        return len(self.subjects)

    def get_data(self, idx, thr=0.7):
        # data with minute resolution
        # thr = threshold for cleaning the signal from artifacts
        where_to_save = self.pth/'data_minute'
        where_to_save.mkdir(parents=True,exist_ok=True)
        where_to_save = where_to_save/(self.subjects.loc[idx,'id'] + '.minute')

        if not( where_to_save.is_file() ):
            data = list()
            for i,rec in enumerate(self.subjects.loc[idx,'rec_path']):
                df = pd.read_csv(rec,sep=';', parse_dates=['Date'], skiprows=1 )
                data.append(df)

            # concatenate recording
            data = pd.concat(data)
            data = data.groupby('Date',sort=False).mean()
            idx = pd.date_range(data.index[0].floor('H'), data.index[-1].ceil('H'), freq='1S')
            data = data.reindex(idx, fill_value=np.nan).reset_index().rename(columns={'index':'Date'})
            data['isDay'] = data['isDay'].fillna(method='ffill')
            data['hour'] = data['Date'].dt.hour
            data['minute'] = (data['Date'].dt.hour*60) + data['Date'].dt.minute
            data[['RT','temp_avg', 'temp_med', 'temp_max', 'centroid_x', 'centroid_y']] = data[['RT','temp_avg', 'temp_med', 'temp_max', 'centroid_x', 'centroid_y']].interpolate()
            data['distance'] = self.euclid(data).fillna(0)

            data['start_date'] = data['Date'].min()
            data['day'] = (data['Date'] - data['start_date']).dt.days + 1

            # recording in minutes
            data_min = data.groupby(['minute','day'], sort=False).mean()
            data_min['temp_rt_diff'] = data_min['temp_avg']-data_min['RT']
            data_min['temp_norm'] = data_min['temp_avg']-data_min['temp_avg'].mean()
            data_min['RT_norm'] = data_min['RT']-data_min['RT'].mean()
            data_min['temp_rt_corrected'] = data_min['temp_norm']-data_min['RT_norm']
            data_min.loc[data_min['temp_rt_corrected'].diff().abs()>thr, 'temp_rt_corrected']=np.nan
            data_min['temp_rt_corrected'] = data_min['temp_rt_corrected'].interpolate()
            data_min = data_min.drop(['ID','timeStamp'],axis=1).reset_index()
            data_min.to_csv(where_to_save.as_posix(),sep=';')
        else:
            data_min = pd.read_csv(where_to_save.as_posix(),sep=';',index_col=0)

        return data_min

    def get_day_avg(self, idx, sort=False, thr=0.7):
        # data with minute resolution (24h [1440 mins] average)
        where_to_save = self.pth/'data_day'
        where_to_save.mkdir(parents=True,exist_ok=True)
        where_to_save = where_to_save/(self.subjects.loc[idx,'id']+'.day')

        if not( where_to_save.is_file() ):

            data = self.get_data(idx,thr=thr)
            # average day
            data_day = data.groupby(['minute'],sort=False).mean().drop('day', axis=1).reset_index()
            if sort:
                data_day = data_day.sort_values(by='minute').reset_index(drop=True)
            data_day.to_csv(where_to_save.as_posix(),sep=';')
        else:
            data_day = pd.read_csv(where_to_save.as_posix(), sep=';', index_col=0)
            if sort:
                data_day = data_day.sort_values(by='minute').reset_index(drop=True)

        return data_day

    def get_single_day(self, idx, day=1, thr=0.7):
        data_min = self.get_data(idx,thr=thr)
        i = 1440 * day
        single_day = data_min.loc[0+i:i+1439]
        return single_day

    def number_of_days(self,idx):
        data_min = self.get_data(idx)
        return data_min.day.max()-1

    def iter_data(self):
        for i in self.subjects.index:
            yield self.subjects.loc[i], self.get_data(i)
    def iter_day_avg(self):
        for i in self.subjects.index:
            yield self.subjects.loc[i], self.get_day_avg(i)

    def iter_single_days(self,idx):
        for d in self.number_of_days(idx):
            yield self.get_single_day(idx,d)


class Cosinor:


    def __init__(self, input_recording, timeHours= None, fs=60, isDay=None):
        self.signal = input_recording
        self.fs = fs

        if timeHours is None:
            self.time = np.linspace(0,1,len(self.signal)) * (len(self.signal)/self.fs)
        else:
            self.time = timeHours

        self.power_spectrum, self.period, self.peaks_signi = self.periodogram(self.signal,self.fs)
        # fit params
        self.fit24()

    def fitComponents(self):
        estim_mesor = np.mean(self.signal)
        estim_acrophase = self.time[np.argmax(self.signal)]
        self.p0 = [estim_mesor]

        self.bounds = [[-np.inf],[np.inf]]
        for i,comp in self.peaks_signi.iterrows():
            self.p0.append( comp['amplitude'])
            self.bounds[0].append(0)
            self.bounds[1].append(np.inf)
            self.p0.append( comp['period'])
            self.bounds[0].append(0)
            self.bounds[1].append(200)
            self.p0.append( estim_acrophase )
            self.bounds[0].append(0)
            self.bounds[1].append( np.inf) #self.time[-1])

        self.params, _ = curve_fit(self.multicomponent_cosinor, self.time, self.signal, p0=self.p0, bounds=self.bounds)

        self.mesor = self.params[0]

        self.compontents = dict()

        inc = 1
        for c in range(0,len(self.peaks_signi)):
            if inc==1:
                self.compontents['amplitude'] = [self.params[inc]]
                self.compontents['period'] = [self.params[inc+1]]
                self.compontents['period_r'] = [np.round(self.params[inc+1])]
                self.compontents['acrophase'] = [np.round( self.params[inc+2] % self.params[inc+1], 3 )]
            else:
                self.compontents['amplitude'].append( self.params[inc] )
                self.compontents['period'].append( self.params[inc+1] )
                self.compontents['period_r'].append( np.round(self.params[inc+1]) )
                self.compontents['acrophase'].append( np.round( self.params[inc+2] % self.params[inc+1], 3 ) )
            inc=inc+3

        self.compontents = pd.DataFrame(self.compontents)[['period','period_r','amplitude','acrophase']]
        self.compontents = self.compontents.sort_values('amplitude',ascending=False).reset_index(drop=True)
        self.curve = self.multicomponent_cosinor(self.time, *self.params)

    def fit24(self):
        estim_mesor = np.mean(self.signal)
        estim_amplitude = np.max(self.signal)-np.min(self.signal)
        estim_acrophase = self.time[np.argmax(self.signal)]
        self.p0 = [estim_mesor,estim_amplitude,24,estim_acrophase]

        self.bounds = [[-np.inf,0,0,0], [np.inf,np.inf,200,np.inf]]

        self.params, _ = curve_fit(self.multicomponent_cosinor, self.time, self.signal, p0=self.p0, bounds=self.bounds)
        self.compontents = dict()
        self.compontents['mesor'] = self.params[0]
        self.compontents['amplitude'] = [self.params[1]]
        self.compontents['period'] = [self.params[2]]
        self.compontents['acrophase'] = [np.round( self.params[3] % 24, 3 )]

        self.compontents = pd.DataFrame(self.compontents)
        self.curve = self.multicomponent_cosinor(self.time, *self.params)

    def fitComponent(self,component=24):
        estim_mesor = np.mean(self.signal)
        estim_acrophase = self.time[np.argmax(self.signal)]
        estim_amplitude = np.max(self.signal)-np.min(self.signal)
        p0 = [estim_mesor,estim_amplitude,component,estim_acrophase]
        print('p0', p0)
        bounds = [[-np.inf,0,0,0], [np.inf,np.inf,200, self.time[-1]*2]]
        params, _ = curve_fit(self.multicomponent_cosinor, self.time, self.signal, p0=p0, bounds=bounds)
        component = dict()
        component['mesor'] = params[0]
        component['amplitude'] = [params[1]]
        component['period'] = [params[2]]
        component['acrophase'] = [np.round( params[3] % 24, 3 )]

        component = pd.DataFrame(component)[['period','amplitude','acrophase']]
        curve = self.multicomponent_cosinor(self.time, *params)

        return component,curve,self.time

    @staticmethod
    def multicomponent_cosinor(x, mesor, *args):
        num_components = len(args) // 3
        result = np.full_like(x, mesor)

        for i in range(num_components):
            amplitude = args[3 * i]
            period = args[3 * i + 1]
            acrophase = args[3 * i + 2]
            result += amplitude * np.cos((2 * np.pi * (x - acrophase)) / period)

        return result

    @staticmethod
    def periodogram(x,fs=60):
        frequencies, power_spectrum = signal.periodogram(x, fs)
        frequencies = frequencies
        power_spectrum =  np.insert(np.flip(power_spectrum[1:]),0,power_spectrum[0])
        period =  np.insert(np.flip(1 / frequencies[1:]),0,0)

        p_values = 1 - power_spectrum  # Convert power values to p-values
        rejected, _ = fdrcorrection(p_values)

        peaks = period[rejected]
        psd = power_spectrum[rejected]
        amplitudes = np.sqrt(power_spectrum[rejected])
        peaks_signi = pd.DataFrame({'period': peaks,'period_r': np.round(peaks), 'psd': psd ,'amplitude': amplitudes})
        peaks_signi = peaks_signi.sort_values('amplitude',ascending=False).reset_index(drop=True)
        return power_spectrum, period, peaks_signi