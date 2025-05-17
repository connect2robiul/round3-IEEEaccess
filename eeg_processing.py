import os
import numpy as np
import scipy.signal as signal
from config import Config
import streamlit as st


class EEG:
    def __init__(self, participant, dataset):
        self.int = {}
        self.delta = {}
        self.theta = {}
        self.alpha = {}
        self.beta = {}
        self.gamma = {}

        base = "/mount/src/round3-ieeeaccess/datasets"
        if dataset == "Mona Lisa":
            Dname = "/Figs for spectra"
            var1 = "/Backgr_int_"
            var2 = ".dat"
        else:
            Dname = "/Cubes for spectra"
            var1 = "/Backgr_int_"
            var2 = "_type_0.4.dat"

        for intensity in Config.INTENSITIES:
            S = os.path.join(base, "Participant " + str(participant) + Dname + var1 + str(intensity) + var2)
            st.write(S)
         #   abs_path = os.path.abspath(S)
          #  st.write("Come from EEG Process:Loading EEG file from:", abs_path)

            I = intensity
            self.int[I] = np.loadtxt(S)

            st.write("ALL good")

            delta_signal = np.empty(self.int[I].shape)
            theta_signal = np.empty(self.int[I].shape)
            alpha_signal = np.empty(self.int[I].shape)
            beta_signal = np.empty(self.int[I].shape)
            gamma_signal = np.empty(self.int[I].shape)

            for c in range(self.int[I].shape[1]):
                delta_signal[:, c], theta_signal[:, c], alpha_signal[:, c], \
                    beta_signal[:, c], gamma_signal[:, c] = self.fir_filtering(I, c)

            self.delta[I] = delta_signal
            self.theta[I] = theta_signal
            self.alpha[I] = alpha_signal
            self.beta[I] = beta_signal
            self.gamma[I] = gamma_signal

    def fir_filtering(self, intensity, channel_idx):
        fs = 250
        filter_delta = signal.firwin(400, [1.0, 4.0], pass_zero=False, fs=fs)
        filter_theta = signal.firwin(400, [5.0, 8.0], pass_zero=False, fs=fs)
        filter_alpha = signal.firwin(400, [8.0, 12.0], pass_zero=False, fs=fs)
        filter_beta = signal.firwin(400, [13.0, 30.0], pass_zero=False, fs=fs)
        filter_gamma = signal.firwin(400, [31.0, 45.0], pass_zero=False, fs=fs)

        res_delta = signal.convolve(self.int[intensity][:, channel_idx], filter_delta, mode='same')
        res_theta = signal.convolve(self.int[intensity][:, channel_idx], filter_theta, mode='same')
        res_alpha = signal.convolve(self.int[intensity][:, channel_idx], filter_alpha, mode='same')
        res_beta = signal.convolve(self.int[intensity][:, channel_idx], filter_beta, mode='same')
        res_gamma = signal.convolve(self.int[intensity][:, channel_idx], filter_gamma, mode='same')

        return res_delta, res_theta, res_alpha, res_beta, res_gamma
    


def createdataframe(user,dataset,signal_fiq):
    global Intencity,channel_names
    tem_signal = []
    eeg_person = EEG(user,dataset)
    for i in Intencity:
        for ch in range(0,31):
            tem = [dataset,ch,channel_names[ch],convart_sin_fiq(signal_fiq),signal_fiq,i,user]
            # tem = [channel_names[ch]]
            S = 'tem_signal.append(np.append(tem, eeg_person.'+signal_fiq+'.get(i)[:,ch][0:15000]))'
            eval(S)
    
           
    tem_dataframe = pd.DataFrame(tem_signal)

    colunm = [] 

    colunm.append("Dataset")
    colunm.append("channel")
    colunm.append("Channel_Name")
    colunm.append("band")
    colunm.append("frequency")
    colunm.append("Intencity")
    colunm.append("participant")

    for col in range(0,15000):
      S = 'Time'+str(col)
      colunm.append(S)    
    
    
    
    tem_dataframe.columns  = colunm 
    return tem_dataframe


def add(a, b):
    print(f"a: {a}, b: {b}")  # debug
    return a + b
