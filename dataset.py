#!/usr/bin/python3

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import librosa
from torch.utils.data.dataloader import default_collate
import glob
import random
import numpy
import soundfile
import csv
from pathlib import Path
from scipy import signal
from RawBoost import process_Rawboost_feature


def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


def pad_dataset(wav, audio_length=64600):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = audio_length

    if waveform_len >= cut:
        waveform = waveform[:cut]
    else:
        num_repeats = int(cut / waveform_len) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    return waveform


class AudioAugmentor:
    def __init__(self, rir_path='yourrir/RIRS_NOISES', musan_path='yourmusan'):
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = self._load_noiselist(musan_path)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))

    def _load_noiselist(self, musan_path):
        noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            category = file.split('/')[-3]
            if category not in noiselist:
                noiselist[category] = []
            noiselist[category].append(file)
        return noiselist

    def add_rev(self, audio, audio_length):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :audio_length]

    def add_noise(self, audio, noisecat, audio_length):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []

        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = audio_length
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)

        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class atadd_dataset(Dataset):
    def __init__(self, path_to_audio, path_to_protocol,
                 rawboost=False, musanrir=False, audio_length=64600):
        super(atadd_dataset, self).__init__()

        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"fake": 1, "real": 0}
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor()

        self.all_files = []
        with open(self.path_to_protocol, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["name"].strip()
                label = row["label"].strip()
                self.all_files.append((filename, label))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)

        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr=sr)

        waveform = pad_dataset(waveform, self.audio_length)

        if self.musanrir:
            audio_length = waveform.size(0)
            waveform = self._apply_augmentation(waveform, audio_length)

        label = self.label[label]
        return waveform, filename, label

    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)

        if augtype == 0:
            return waveform
        elif augtype == 1:
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        elif augtype in [2, 3, 4]:
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform

        return waveform

    def collate_fn(self, samples):
        return default_collate(samples)


class ProtocolManifestDataset(Dataset):
    """Read a continual-learning manifest with absolute or relative paths.

    When ``RFPROMPT_DATA_ROOT`` is set, relative paths are resolved from that
    directory. Otherwise, they are resolved from the manifest directory.
    """

    def __init__(self, manifest_path, audio_length=64600):
        super(ProtocolManifestDataset, self).__init__()
        self.manifest_path = manifest_path
        self.audio_length = audio_length
        self.all_files = []

        with open(manifest_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            required = {'utt_id', 'audio_path', 'label'}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
                )
            for row in reader:
                audio_path = Path(row['audio_path'].strip())
                if not audio_path.is_absolute():
                    data_root = os.environ.get('RFPROMPT_DATA_ROOT')
                    if data_root:
                        audio_path = Path(data_root).expanduser() / audio_path
                    else:
                        audio_path = Path(manifest_path).resolve().parent / audio_path
                self.all_files.append((
                    str(audio_path),
                    row['utt_id'].strip(),
                    int(row['label'])
                ))

        if not self.all_files:
            raise ValueError(f"Manifest is empty: {manifest_path}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        audio_path, utt_id, label = self.all_files[idx]
        waveform, _ = torchaudio_load(audio_path)
        waveform = pad_dataset(waveform, self.audio_length)
        return waveform, utt_id, label
