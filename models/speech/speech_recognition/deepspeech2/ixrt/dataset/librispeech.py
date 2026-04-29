"""
Define the dataset of LibriSpeech
"""

import os
import glob
import soundfile
import numpy as np


class LibriSpeech:
    def __init__(self, dataroot, transform=None):
        assert os.path.exists(dataroot), f"The {dataroot} must be existed!"
        self.dataroot = dataroot
        self._parse_file()
        self.transform = transform

    def _parse_file(self):
        """
        Parse the test-clean data and groundtruth
        """
        audio_names = []
        text_transcripts = []
        text_pattern = os.path.join(self.dataroot, "test-clean", "*", "*", "*.trans.txt")
        text_files = glob.glob(text_pattern)
        print(f"[INFO]: text files length: {len(text_files)}")

        for text_file in text_files:
            # print(f"Processing: {os.path.basename(text_file)}")

            with open(text_file, 'r') as f:
                lines = f.readlines()

            lines = [line.strip().split(' ', maxsplit=1) for line in lines]
            for line in lines:
                audio_name, text = line
                audio_names.append(audio_name)
                text_transcripts.append(text)

        self.audio_names = audio_names
        self.text_transcripts = text_transcripts
        print("[INFO]: Achieve Parsing!")

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):

        audio_name = self.audio_names[idx]
        text_gt = self.text_transcripts[idx]

        # print(f"audio_name: {audio_name}")
        # print(f"text_gt: {text_gt}")

        name, subname, _ = audio_name.split('-')
        audio_file = os.path.join(self.dataroot, "test-clean", name, subname, audio_name + ".flac")

        audio, sample_rate = soundfile.read(audio_file, dtype="int16", always_2d=True)
        audio = audio[:, 0]
        # print(f"audio shape: {audio.shape}")

        if self.transform is None:
            input_data = audio
        else:
            preprocess_args = {"train": False}
            input_data = self.transform(audio, **preprocess_args)

        input_data = np.expand_dims(input_data.astype(np.float32), axis=0)
        return input_data, text_gt

