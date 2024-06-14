from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import os
import subprocess
import torchaudio
import soundfile as sf
import numpy as np
import torch

class audioDenoise:
  '''
    Audio File Denoise Flow
  '''
  def __init__(self, 
               use_spleeter=True, 
               use_MVSEP=False,
               use_deepfiller3=False,
               output_sampling_rate=16000):
    self.use_spleeter=use_spleeter
    self.use_MVSEP=use_MVSEP
    self.use_deepfiller3=use_deepfiller3

    def spleeter_phrase(self, audio_path, output_folder):
      file_name=os.path.basename(audio_path)
      subprocess.run(["spleeter", "separate", "-o", output_folder, audio_path])
      return output_folder+f'{file_name}/vocals.wav'
    
    def mvSEP_phrase(self, input_path, output_folder):
      command = [
        'python', './MVSEP-MDX23-Colab_v2/inference.py',
        '--input_audio', input_path,
        '--large_gpu',
        '--BSRoformer_model', 'ep_317_1297',
        '--weight_BSRoformer', '10',
        '--weight_InstVoc', '4',
        '--weight_InstHQ4', '2',
        '--weight_VOCFT', '2',
        '--weight_VitLarge', '1',
        '--overlap_demucs', '0.6',
        '--overlap_VOCFT', '0.1',
        '--overlap_InstHQ4', '0.1',
        '--output_format', 'FLOAT',
        '--BigShifts', '3',
        '--output_folder', output_folder,
        '--input_gain', '0',
        '--filter_vocals',
        '--restore_gain',
        '--vocals_only',
        '--use_VitLarge',
        '--use_VOCFT',
        '--use_InstHQ4',
        '--use_InstVoc',
        '--use_BSRoformer'
      ]
      subprocess.run(command)
      return output_folder+'/audio_vocals.wav'

    def deepfiller_phrase(self, audio_path, save_path):
      model, df_state, _ = init_df()
      audio, _ = load_audio(audio_path, sr=df_state.sr())
      enhanced = enhance(model, df_state, audio)
      save_audio(save_path, enhanced, df_state.sr())
      return save_path
    
    def resampling_rate_phrase(self, audio_path, save_path, max_seconds=60000):
      batch = {"file": audio_path}
      speech_array, sampling_rate = torchaudio.load(batch["file"])
      if sampling_rate != self.output_sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sampling_rate,
                                                  new_freq=self.output_sampling_rate)
        speech_array = transform(speech_array)
      speech_array = speech_array[0]
      if max_seconds > 0:
        speech_array = speech_array[:max_seconds*self.output_sampling_rate]
      sf.write(save_path, speech_array.numpy(), samplerate=self.output_sampling_rate)
      return save_path
    
    def denoise(audio_path):
      clear_audio_path=audio_path
      if self.use_spleeter: clear_audio_path=self.spleeter_phrase(clear_audio_path, './spleeterClear')

      if self.use_MVSEP: 
        if torch.cuda.device_count()<1: print('NO CUDA FOUND--> SKIP MVSEP phrase')
        else: clear_audio_path=self.mvSEP_phrase(clear_audio_path, './MVSEPCLear')

      if self.use_deepfiller3: clear_audio_path=self.deepfiller_phrase(clear_audio_path, 'deepfillerResult.wav')
      self.resampling_rate_phrase(clear_audio_path, 'clearaudio.wav')
      return 'clearaudio.wav'
