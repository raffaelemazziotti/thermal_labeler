from scipy import signal
from statsmodels.stats.multitest import fdrcorrection
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import ttest_1samp
from sklearn.metrics import r2_score

hour2radians = lambda x: ((x%24)/24)*2*np.pi
radians2hour = lambda x: ((x%(2*np.pi))/(2*np.pi))*24
wrap2pi = lambda phases: (phases + np.pi) % (2 * np.pi) - np.pi


class Subjects():

    def __init__(self, pth=None):
        """
            Initializes an instance of the Subjects class.

            Parameters:
            - pth (str or Path object, optional): Path to the directory containing the thermal data.
              If not provided, the current working directory appended with 'thermal_data' is used.
        """
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
        """
            Computes the Euclidean distance between consecutive centroid coordinates in a DataFrame.

            Parameters:
            - df (pandas.DataFrame): DataFrame containing centroid_x and centroid_y columns.

            Returns:
            - numpy.ndarray: Euclidean distances between consecutive centroid coordinates.
        """
        return np.sqrt((df['centroid_x'] - df['centroid_x'].shift(1))**2 +
                       (df['centroid_y'] - df['centroid_y'].shift(1))**2
                       )

    def __str__(self):
        """
        Returns a string representation of the Subjects object.

        Returns:
        - str: String representation of the Subjects object.
        """
        string = 'Number of Subjects: ' + str( len(self.subjects) ) + '\n\n'
        for i,row in self.subjects.iterrows():
            string = string + str(i) + ' -> ID: ' + row['id'] + ' RECs: ' + str( row['rec_n'] ) + '\n'
        return string

    def __getitem__(self,idx):
        """
        Returns the subject at the specified index.

        Parameters:
        - idx (int): Index of the subject.

        Returns:
        - pandas.Series: Information about the subject at the specified index.
        """
        return self.subjects.loc[idx]

    def __len__(self):
        """
        Returns the number of subjects in the Subjects object.

        Returns:
        - int: Number of subjects.
        """
        return len(self.subjects)

    @staticmethod
    def remove_outliers(signal, threshold_multiplier=2):
        diff = np.diff(signal)  # Calculate the first differences of the signal
        abs_diff = np.abs(diff)  # Take the absolute value of the differences
        median = np.median(abs_diff)
        mad = np.median(np.abs(abs_diff - median))
        threshold = threshold_multiplier * mad

        cleaned_signal = signal.copy()
        outliers = abs_diff > threshold

        # Replace outliers with NaN
        cleaned_signal[1:][outliers] = np.nan

        # Perform linear interpolation to fill in NaN values
        cleaned_signal = pd.Series(cleaned_signal).interpolate().values

        return cleaned_signal, threshold

    def get_data(self, idx):
        """
        Retrieves data for a subject at the specified index.

        Parameters:
        - idx (int): Index of the subject.

        Returns:
        - pandas.DataFrame: Data for the specified subject, cleaned and processed.
        """

        # data with minute resolution
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
            data_min['temp_avg'] = self.remove_outliers(data_min['temp_avg'])[0]
            data_min['RT'] = self.remove_outliers(data_min['RT'])[0]
            data_min['temp_rt_diff'] = data_min['temp_avg']-data_min['RT']
            data_min['temp_norm'] = data_min['temp_avg']-data_min['temp_avg'].mean()
            data_min['RT_norm'] = data_min['RT']-data_min['RT'].mean()
            data_min['temp_rt_corrected'] = data_min['temp_norm']-data_min['RT_norm']
            data_min['temp_rt_corrected'] = data_min['temp_rt_corrected'].interpolate()
            data_min = data_min.drop(['ID','timeStamp'],axis=1).reset_index()
            data_min.to_csv(where_to_save.as_posix(),sep=';')
        else:
            data_min = pd.read_csv(where_to_save.as_posix(),sep=';',index_col=0)

        return data_min

    def get_day_avg(self, idx, sort=False):
        """
        Retrieves the daily average data for a subject at the specified index.

        Parameters:
        - idx (int): Index of the subject.
        - sort (bool, optional): Whether to sort the data by minute.

        Returns:
        - pandas.DataFrame: Daily average data for the specified subject.

        ##### EXAMPLES
        # Create an instance of the Subjects class
        subjects = Subjects()

        # Retrieve the number of subjects
        num_subjects = len(subjects)
        print("Number of subjects:", num_subjects)

        # Access a subject at a specific index
        subject = subjects[0]
        print("Subject at index 0:", subject)

        # Retrieve data for a subject
        data = subjects.get_data(0)
        print("Data for subject at index 0:")
        print(data.head())

        # Retrieve daily average data for a subject
        daily_avg_data = subjects.get_day_avg(0)
        print("Daily average data for subject at index 0:")
        print(daily_avg_data.head())

        # Retrieve data for a single day of a subject
        single_day_data = subjects.get_single_day(0, day=2)
        print("Data for day 2 of subject at index 0:")
        print(single_day_data.head())

        # Compute the number of days for a subject
        num_days = subjects.number_of_days(0)
        print("Number of days for subject at index 0:", num_days)

        # Iterate over subjects and their data
        for subject, data in subjects.iter_data():
            print("Subject:", subject)
            print("Data:")
            print(data.head())

        # Iterate over subjects and their daily average data
        for subject, daily_avg_data in subjects.iter_day_avg():
            print("Subject:", subject)
            print("Daily Average Data:")
            print(daily_avg_data.head())

        # Iterate over days of a specific subject
        for day, data in subjects.iter_single_days(0):
            print("Day:", day)
            print("Data:")
            print(data.head())
        """
        # data with minute resolution (24h [1440 mins] average)
        where_to_save = self.pth/'data_day'
        where_to_save.mkdir(parents=True,exist_ok=True)
        where_to_save = where_to_save/(self.subjects.loc[idx,'id']+'.day')

        if not( where_to_save.is_file() ):

            data = self.get_data(idx)
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

    def get_single_day(self, idx, day=0):
        """
        Retrieves the data for a single day of a subject at the specified index.

        Parameters:
        - idx (int): Index of the subject.
        - day (int, optional): Index of the day (0-indexed).

        Returns:
        - pandas.DataFrame: Data for the specified day of the subject.
        """
        data_min = self.get_data(idx)
        i = 1440 * day
        single_day = data_min.loc[0+i:i+1439]
        return single_day

    def number_of_days(self,idx):
        """
        Computes the number of days for which data is available for a subject at the specified index.

        Parameters:
        - idx (int): Index of the subject.

        Returns:
        - int: Number of days for the specified subject.
        """
        data_min = self.get_data(idx)
        return data_min.day.max()-1

    def iter_data(self):
        """
        Iterates over the subjects and their corresponding data.

        Yields:
        - Tuple[pandas.Series, pandas.DataFrame]: Tuple containing the subject information and their data.
        """
        for i in self.subjects.index:
            yield self.subjects.loc[i], self.get_data(i)

    def iter_day_avg(self):
        """
        Iterates over the subjects and their corresponding daily average data.

        Yields:
        - Tuple[pandas.Series, pandas.DataFrame]: Tuple containing the subject information and their daily average data.
        """
        for i in self.subjects.index:
            yield self.subjects.loc[i], self.get_day_avg(i)

    def iter_single_days(self,idx):
        """
           Iterates over the days of a specific subject, yielding the day index and the corresponding data.

           Parameters:
           - idx (int): Index of the subject.

           Yields:
           - Tuple[int, pandas.DataFrame]: Tuple containing the day index and the
           for d in range(self.number_of_days(idx) ):
               yield d,self.get_single_day(idx,d)
           """
        for d in range(self.number_of_days(idx) ):
            yield d,self.get_single_day(idx,d)

    def get_days_df(self,idx, what='temp_rt_corrected'):
        """
        Retrieve a DataFrame containing data for each individual day.

        Parameters:
            idx (int): Index of the day to retrieve data for.
            what (str, optional): Name of the column to include in the DataFrame. Default is 'temp_rt_corrected'.

        Returns:
            each_day (pd.DataFrame): DataFrame containing data for each individual day.
        """

        each_day = dict()
        for d,day in self.iter_single_days(idx):
            each_day[d] = day[what].reset_index(drop=True)
        each_day = pd.DataFrame.from_dict(each_day)
        return each_day

class Periodogram:

    def __init__(self, signal_input, fs=60):
        """
         Initialize the Periodogram object.

         Args:
             signal_input (array-like): Input signal for which the periodogram will be computed.
             fs (float, optional): Sampling frequency of the input signal. Defaults to 60.

             # EXAMPLES
             time = np.linspace(0, 10, 1000)
             frequency = 2  # Hz
             amplitude = 1
             signal_input = amplitude * np.sin(2 * np.pi * frequency * time)

             # Create a Periodogram object
             periodogram = Periodogram(signal_input, fs=100)  # Sampling frequency is 100 Hz

             # Get the average power spectral density for specific periods
             periods = [2, 4, 6]
             average_psd = periodogram.get_psd(periods)
             print("Average PSD:", average_psd)

             # Plot the power spectrum with significant peaks
             periodogram.plot(signi=True)
         """

        #signal_input = signal_input * np.hanning(len(signal_input))
        frequencies, power_spectrum = signal.periodogram(signal_input, fs)
        self.power_spectrum =  np.insert(np.flip(power_spectrum[1:]),0,power_spectrum[0])
        self.period =  np.insert(np.flip(1 / frequencies[1:]),0,0)

        poi = np.where((self.period>0.8) & (self.period<=72) )[0]
        self.period = self.period[poi]
        self.power_spectrum = self.power_spectrum[poi]
        self.interp_func = interp1d(self.period, self.power_spectrum, kind='linear')

        significant_indices, _ = fdrcorrection( 1 - self.power_spectrum )

        peaks = self.period[significant_indices]
        psd = self.power_spectrum[significant_indices]
        amplitudes = np.sqrt(self.power_spectrum[significant_indices])
        self.peaks_signi = pd.DataFrame({'period': peaks,'period_r': np.round(peaks), 'psd': psd ,'amplitude': amplitudes})
        self.peaks_signi = self.peaks_signi.sort_values('amplitude',ascending=False).reset_index(drop=True)

    def get_psd(self, periods):
        """
         Get the average power spectral density (PSD) for the given periods.

         Args:
             periods (array-like): Periods for which to compute the average PSD.

         Returns:
             float: Average PSD for the given periods.
         """
        return np.mean(self.interp_func(periods))

    def plot(self, ax=None, signi=True, to_amp=False):
        """
         Plot the power spectrum.

         Args:
             ax (matplotlib.axes.Axes, optional): Axes object for the plot. If not provided, a new figure and axes will be created. Defaults to None.
             signi (bool, optional): Whether to show significant peaks. Defaults to True.
             to_amp (bool, optional): Whether to plot the amplitude (square root of PSD) instead of PSD. Defaults to False.

         Returns:
             matplotlib.axes.Axes: Axes object containing the plot.
         """

        if ax is None:
            plot_labels=True
            fig, ax = plt.subplots(figsize=(6,4))
        else:
            plot_labels=False

        if to_amp:
            y = np.sqrt(self.power_spectrum)
        else:
            y = self.power_spectrum
        ax.bar( self.period, y)

        if signi:
            if to_amp:
                star_pos = np.sqrt( self.peaks_signi['psd'].max() ) * 0.05
            else:
                star_pos = self.peaks_signi['psd'].max() *0.05

            for i,row in self.peaks_signi.iterrows():
                if to_amp:
                    y = row['amplitude']
                else:
                    y = row['psd']
                ax.plot(row['period'],y+star_pos,'r*')

        if plot_labels:
            ax.set_xlabel('Period [Hours]')
            if to_amp:
                ax.set_ylabel('Magnitude')
            else:
                ax.set_ylabel('PSD')

        return ax

class Cosinor:

    def __init__(self, input_recording, timeHours= None, fs=60, fixed24=True, acroWrap=False):
        """
         Initialize a Cosinor object.

         Parameters:
             input_recording (array-like): The input signal.
             timeHours (array-like, optional): The time points corresponding to the input signal in hours. If not provided, the time points are calculated based on the signal length and the sampling frequency.
             fs (int, optional): The sampling frequency in Hz. Default is 60.
             fixed24 (bool,optional): force 24h period
             acroWrap (bool,optional): force acrophase to 0 24h

         """
        self.signal = input_recording
        self.fs = fs

        if timeHours is None:
            self.time = np.linspace(0,1,len(self.signal)) * (len(self.signal)/self.fs)
        else:
            self.time = timeHours

        self.fit24(fixed=fixed24, acroWrap=acroWrap)

    def fitComponents(self, input_components, fixed=True):
        """
         Fit multiple components to the input signal.

         Parameters:
             components (array-like): The periods of the components to fit.

         Returns:
             None
         """
        estim_mesor = np.mean(self.signal)
        estim_acrophase = self.time[np.argmax(self.signal)]
        p0 = [estim_mesor]

        bounds = [[-np.inf],[np.inf]]
        for comp in input_components:
            p0.append( self.component_amp(self.signal, comp, self.fs) )
            bounds[0].append(0)
            bounds[1].append( np.inf )
            p0.append( comp )
            if fixed:
                bounds[0].append(comp-0.01)
                bounds[1].append(comp+0.01)
            else:
                bounds[0].append(0)
                bounds[1].append(200)
            p0.append( estim_acrophase )
            bounds[0].append(0)
            bounds[1].append( np.inf )

        params, _ = curve_fit(self.multicomponent_cosinor, self.time, self.signal, p0=p0, bounds=bounds)

        mesor = params[0]

        components = dict()

        inc = 1
        for c in range(0,len(input_components)):
            if inc==1:
                components['amplitude'] = [params[inc]]
                components['period'] = [params[inc+1]]
                components['period_r'] = [np.round(params[inc+1])]
                components['acrophase'] = [np.round( params[inc+2] % params[inc+1], 3 )]
            else:
                components['amplitude'].append( params[inc] )
                components['period'].append( params[inc+1] )
                components['period_r'].append( np.round(params[inc+1]) )
                components['acrophase'].append( np.round( params[inc+2] % params[inc+1], 3 ) )
            inc=inc+3

        print(components)
        components = pd.DataFrame(components)[['period','period_r','amplitude','acrophase']]
        components = components.sort_values('amplitude',ascending=False).reset_index(drop=True)
        curve = self.multicomponent_cosinor(self.time, *params)

        return mesor, components, curve, self.time

    def fit24(self, fixed=False, acroWrap=False):
        """
        Fit a 24-hour component to the input signal.

        Parameters:
            fixed (bool, optional): Whether to fix the period to 24 hours. If True, the period will be fixed, otherwise it will be optimized. Default is False.
            acroWrap (bool,optional): force acrophase to 0 24h
        Returns:
            None
        """
        estim_mesor = np.mean(self.signal)
        estim_amplitude = np.max(self.signal)-np.min(self.signal)
        estim_acrophase = self.time[np.argmax(self.signal)]
        self.p0 = [estim_mesor,estim_amplitude,24,estim_acrophase]

        if fixed:
            self.bounds = [[-np.inf,0,23.99,0], [np.inf,np.inf,24.01,np.inf]]
        else:
            self.bounds = [[-np.inf,0,0,0], [np.inf,np.inf,200,np.inf]]

        self.params, _ = curve_fit(self.multicomponent_cosinor, self.time, self.signal, p0=self.p0, bounds=self.bounds)
        self.components = dict()
        self.components['mesor'] = self.params[0]
        self.components['amplitude'] = [self.params[1]]
        self.components['period'] = [self.params[2]]
        if acroWrap:
            self.components['acrophase'] = [np.round( self.params[3] % 24, 3 )]
        else:
            self.components['acrophase'] = [np.round( self.params[3], 3 )]

        self.curve = self.multicomponent_cosinor(self.time, *self.params)
        self.components['r2'] = self.calculate_r2_score(self.signal, self.curve)

        self.components = pd.DataFrame(self.components)


    def fitComponent(self,component=24, fixed=False, acrowrap=True):
        """
         Fit a specific component to the input signal.

         Parameters:
             component (float, optional): The period of the component to fit. Default is 24.
            acroWrap (bool,optional): force acrophase to 0 24h
         Returns:
             tuple: A tuple containing the component's information (amplitude, period, acrophase), the fitted curve, and the time points.
         """
        estim_mesor = np.mean(self.signal)
        estim_acrophase = self.time[np.argmax(self.signal)]
        estim_amplitude = self.component_amp(self.signal, component, self.fs) #np.max(self.signal)-np.min(self.signal)
        p0 = [estim_mesor,estim_amplitude,component,estim_acrophase]

        if fixed:
            bounds = [[-np.inf,0,component-0.01,0], [np.inf,np.inf,component+0.01, self.time[-1]*2]]
        else:
            bounds = [[-np.inf,0,0,0], [np.inf,np.inf,200, self.time[-1]*2]]

        params, _ = curve_fit(self.multicomponent_cosinor, self.time, self.signal, p0=p0, bounds=bounds)
        component = dict()
        component['mesor'] = params[0]
        component['amplitude'] = [params[1]]
        component['period'] = [params[2]]
        if acrowrap:
            component['acrophase'] = [np.round( params[3] % 24, 3 )]
        else:
            component['acrophase'] = [np.round( params[3], 3 )]

        curve = self.multicomponent_cosinor(self.time, *params)
        component['r2'] = self.calculate_r2_score(self.signal, curve)
        component = pd.DataFrame(component)[['mesor','period','amplitude','acrophase','r2']]

        return component,curve,self.time

    @staticmethod
    def multicomponent_cosinor(x, mesor, *args):
        """
         Calculate the multicomponent cosinor curve.

         Parameters:
             x (array-like): The time points.
             mesor (float): The mesor value.
             args (array-like): The parameters of the components (amplitude, period, acrophase).

         Returns:
             array-like: The calculated cosinor curve.
         """
        num_components = len(args) // 3
        result = np.full_like(x, mesor)

        for i in range(num_components):
            amplitude = args[3 * i]
            period = args[3 * i + 1]
            acrophase = args[3 * i + 2]
            result += amplitude * np.cos((2 * np.pi * (x - acrophase)) / period)

        return result

    @staticmethod
    def component_amp(input_signal, target_period, sampling_frequency):
        """
         Calculate the amplitude of a component using periodogram analysis.

         Parameters:
             input_signal (array-like): The input signal.
             target_period (float): The period of the component.
             sampling_frequency (int): The sampling frequency in Hz.

         Returns:
             float: The amplitude of the component.
         """
        frequencies, psd = signal.periodogram(input_signal, fs=sampling_frequency)
        interpolation_func = interp1d(frequencies, psd, kind='linear')
        psd = interpolation_func(1/target_period)
        return np.sqrt( psd  )
    @staticmethod
    def calculate_r2_score(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        return r2


class CrossCorrelation:

    def __init__(self, signal1, signal2, max_latency, resample=100):

        """
        Initialize a CrossCorrelation object.

        Parameters:
            signal1 (array-like): The first input signal.
            signal2 (array-like): The second input signal.
            max_latency (int): The maximum latency in minutes.
            resample (int, optional): The number of resamples for calculating the average cross-correlation. Default is 100.

        Returns:
            None

        # positive latency means that signal1 is lagging
        # negative latency means that signal1 is leading
        """

        self.signal1 = signal1
        self.signal2 = signal2
        self.cross_correlation, self.latencies = self.xcorr(self.signal1, self.signal2, max_latency)
        self.correlation_coeff = np.corrcoef(self.signal1, self.signal2)[0,1]

        if resample:
            per_xcrorr = np.zeros(( resample, len(self.cross_correlation) ) )
            for i in range(0,resample):
                per_sig = np.random.permutation(self.signal2)
                per_xcrorr[i], _ = self.xcorr( self.signal1, per_sig, max_latency)
        self.per_xcorr_avg = np.mean(per_xcrorr,axis=0)


        peak_pos = np.argmax(self.cross_correlation)
        peak_max = self.cross_correlation[peak_pos]
        peak_lat = self.latencies[peak_pos]
        if peak_lat>0:
            peak_sig1_is = 'lag'
        elif peak_lat<0:
            peak_sig1_is = 'lead'
        else:
            peak_sig1_is = 'equal'

        distr = per_xcrorr[:, peak_pos]
        t_statistic, p_val = ttest_1samp(distr,peak_max)
        self.summary = pd.DataFrame([[self.correlation_coeff,peak_pos,peak_max,peak_lat,peak_sig1_is,p_val]],columns=['corr_coeff','latency_sample','amplitude','latency_minute','signal_1_is','p-val'],index=[0])

    @staticmethod
    def xcorr(sig1,sig2,lat_max):
        """
        Calculate the cross-correlation between two signals.

        Parameters:
            sig1 (array-like): The first input signal.
            sig2 (array-like): The second input signal.
            lat_max (int): The maximum latency.

        Returns:
            tuple: A tuple containing the cross-correlation array and the latency array.
        """
        cross_correlation = np.zeros(2 * lat_max + 1)
        latencies = np.arange(-lat_max, lat_max + 1)

        for index, latency in enumerate(latencies):
            shifted_signal = np.roll(sig2, latency)
            cross_correlation[index] = np.correlate(sig1, shifted_signal, mode='valid') / (np.linalg.norm(sig1) * np.linalg.norm(shifted_signal))

        return cross_correlation, latencies

    def plot(self, ax=None, show_perm=True, show_peak=True, line_color='b',line_style='-'):
        """
            Plot the cross-correlation curve.

            Parameters:
                ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
                show_perm (bool, optional): Whether to show the permuted cross-correlation curve. Default is True.
                show_peak (bool, optional): Whether to show the peak of the cross-correlation curve. Default is True.
                line_color (str, optional): The color of the cross-correlation curve. Default is 'b'.
                line_style (str, optional): The line style of the cross-correlation curve. Default is '-'.

            Returns:
                None
        """

        if ax is None:
            plot_labels=True
            fig, ax = plt.subplots(figsize=(4,4))
        else:
            plot_labels=False
        ax.axvline(0,color='k',linestyle='--')

        ax.plot(self.latencies, self.cross_correlation,line_color+line_style, label="real")
        if show_perm:
            ax.plot(self.latencies, self.per_xcorr_avg,'gray', label="perm")

        if show_peak:
            ax.plot(self.summary['latency_minute'], self.summary['amplitude'],'sr')

        if plot_labels:
            ax.set_ylabel('Cross-correlation')
            ax.set_xlabel('Time Lag [minutes]')
            plt.legend(loc="best")


        return ax
