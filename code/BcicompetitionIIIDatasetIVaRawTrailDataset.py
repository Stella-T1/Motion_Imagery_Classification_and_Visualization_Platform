from scipy import signal
import scipy.io as sciio
import numpy as np
from torch.utils.data import Dataset
import os


class BcicompetitionIIIDatasetIVaRawTrailDataset(Dataset):
    root_path = ''
    cue_n_times = 350
    data_path = ''

    def __init__(self, root_path, subject_code: str = 'aa', do_normalization=False, selected_ch_indexes=None) -> None:
        super().__init__()
        self.root_path = root_path
        self.do_normalization = do_normalization
        all_eeg, target, ch_names, x_pos, y_pos = self.read_mat(subject_code)
        self.data = all_eeg
        self.target = target
        self.ch_names = ch_names
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.selected_ch_indexes = selected_ch_indexes

    def read_mat(self, subject_code):
        root_path = self.root_path
        cue_n_times = self.cue_n_times

        mat_full_path = os.path.join(
            root_path, f'data_set_IVa_{subject_code}.mat')
        self.data_path = mat_full_path
        mat_dict = sciio.loadmat(mat_full_path)
        true_label_path = os.path.join(
            root_path, f'true_labels_{subject_code}.mat')
        true_label_mat_dict = sciio.loadmat(true_label_path)
        mrk = mat_dict['mrk']
        nfo = mat_dict['nfo']

        cue_onsets = np.squeeze(mrk[0][0][0])
        target = np.squeeze(true_label_mat_dict['true_y']) - 1
        print(target.shape)
        class_names = np.squeeze(mrk[0][0][2])
        class_names = np.array([class_names[0][0], class_names[1][0]])
        x_pos = np.squeeze(nfo[0][0][3])
        y_pos = np.squeeze(nfo[0][0][4])
        ch_names = np.array(
            list(map(lambda x: x[0], np.squeeze(nfo[0][0][2]))))
        eeg = (mat_dict['cnt'] * 0.1) # [T, C]
        eeg = (eeg.T - np.mean(eeg, axis=1)).T  # common average reference
        sos = signal.butter(8, [4, 40], btype="band", fs=100, output='sos')
        eeg = signal.sosfilt(sos, eeg, axis=1)
        all_eeg = None
        for cue_onset in cue_onsets:
            mi_eeg = eeg[cue_onset: cue_onset + cue_n_times]

            if all_eeg is None:
                all_eeg = mi_eeg
            else:
                all_eeg = np.vstack([all_eeg, mi_eeg])
        print(all_eeg.shape)
        return all_eeg, target, ch_names, x_pos, y_pos

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        eeg = self.data[idx*self.cue_n_times: (idx+1)*self.cue_n_times].T
        if self.selected_ch_indexes != None:
            eeg = np.take(eeg, self.selected_ch_indexes, axis=0)
        return {
            'eeg': eeg,
            'target': self.target[idx]
        }
