#from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import numpy as np
import re
import yt_dlp
import os
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import librosa
import json
import wave
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import torchaudio
import soundfile as sf
import subprocess
import shutil
import torch
import logging
##################################################################################

class audio_auto_crawling:

  '''
    Audio Book Data Crawling from Youtube for Automatic Speech Recognition (ASR) task.
      author: Nguyen Minh Quan
    Version 1.0.1
  '''

  def __init__(self, 
               huggingface_token, 
               source_website="https://www.youtube.com/watch?v={}", 
               raw_audio_save_path='raws/',
               result_caption_path='caption.txt',
               result_audio_folder='./result_wav',
               skip_title=[],
               reverse_skip=False,
               return_total_duration=True,
               language='vietnamese',
               ffmpeg_location='/usr/bin/ffmpeg',
               **kwargs):
    self.website_format = source_website # Source website, such as Youtube... other will be update later!
    self.raw_audio_save_path=raw_audio_save_path # raw Audio save Path
    self.skip_title=skip_title # Skip Audio if title contant: [...]
    self.reverse_skip=reverse_skip # Skip Audio if title not contant skip_title: [...]
    self.device ="cuda" if torch.cuda.is_available() else "cpu"
    self.huggingface_token=huggingface_token # Huggingface token to use Pyannote Segmentation
    self.default_sampling_rate=16000 # Defalut Audio sampling rate 
    self.audio_segmentation_pipeline=None
    self.audio_caption_pipeline=None
    self.result_caption_path=result_caption_path # Text file path to store audio caption
    self.result_audio_folder=result_audio_folder # Folder Path to store Audio
    self.language=language # Main language for Whisper.
    self.return_total_duration=return_total_duration # Return total duration at final
    self.count_duration=-1.0 # Total duration checkpoint
    self.encoding="utf-8" # Encoding for writting file
    self.min_duration=0.1 # Skip if audio duration shorter than min duration
    self.min_merge_allow=1.0 # Merge audio with the next one if audio duration shorter than min_merge_allow
    self.ffmpeg_location=ffmpeg_location

  def video_download(self, 
                     id):

    '''
      The function to download video from Youtube by its ID.    
    '''
    url=self.website_format.format(id)
    path=self.raw_audio_save_path
    ydl_opts={}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info_dict = ydl.extract_info(url, download=False)
    video_title = info_dict['title']
    if len(self.skip_title):
      if self.reverse_skip:
        if not any([True if item in video_title else False for item in self.skip_title]): return ''
      else: 
        if not all([True if item in video_title else False for item in self.skip_title]): return ''
    video_name = re.sub('[\\\\/*?:"<>|]', '', video_title)
    name = video_name
    return_path = f'{path}/{id}'
    ydl_opts = {
      'format': 'm4a/bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'continue_dl': True,
        'outtmpl': return_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'geobypass':True,
        'ffmpeg_location': self.ffmpeg_location
  }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(url)
    return return_path+'.wav'

  def audio_spliting(self, 
                     audio_path, 
                     start, 
                     end, 
                     tgt_path):
    
    '''
      The function to cut audio.
    '''

    # file to extract the snippet from
    with wave.open(audio_path, "rb") as infile:
      # get file data
      nchannels = infile.getnchannels()
      sampwidth = infile.getsampwidth()
      framerate = infile.getframerate()
      # set position in wave to start of segment
      infile.setpos(int(start * framerate))
      # extract data
      data = infile.readframes(int((end - start) * framerate))

    # write the extracted data to a new file
    with wave.open(tgt_path, 'w') as outfile:
      outfile.setnchannels(nchannels)
      outfile.setsampwidth(sampwidth)
      outfile.setframerate(framerate)
      outfile.setnframes(int(len(data) / sampwidth))
      outfile.writeframes(data)

  def to_seconds(self, timestr):
    seconds= 0
    milisec = timestr[-3:]
    for part in timestr[:-4].split(':'):
      seconds= seconds*60 + int(part, 10)
    return float(seconds)+float(f'0.{milisec}')

  def init_download(self):

    if (self.device=='cpu'): logging.warning('No GPU found!\nThis program might take more time to run!')
    # Download Pyannote Segmentation 3.0
    logging.info('-Download Pyannote Segmentation 3.0 Model-')
    modelA = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=self.huggingface_token)
    pipelineA = VoiceActivityDetection(segmentation=modelA)
    HYPER_PARAMETERS = {
      # remove speech regions shorter than that many seconds.
      "min_duration_on": 0.0,
      # fill non-speech regions shorter than that many seconds.
      "min_duration_off": 0.0,
    }
    pipelineA.instantiate(HYPER_PARAMETERS)
    pipelineA.to(torch.device(self.device))
    self.audio_segmentation_pipeline=pipelineA

    # Download Whisper-v3
    logging.info('-Download OpenAI Whisper_v3 Model-')
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(self.device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=self.device,
    )
    self.audio_caption_pipeline=pipe

    if not os.path.exists(self.raw_audio_save_path):
      os.mkdir(self.raw_audio_save_path)

    if not os.path.exists(self.result_audio_folder):
      os.mkdir(self.result_audio_folder)

  # Convert audio sampling rate to default 
  def sampling_rate_converter(self, 
                              path, 
                              save_temp_path, 
                              max_seconds=60000):
    batch = {"file": path}
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    if sampling_rate != self.default_sampling_rate:
      transform = torchaudio.transforms.Resample(orig_freq=sampling_rate,
                                                 new_freq=self.default_sampling_rate)
      speech_array = transform(speech_array)
    speech_array = speech_array[0]
    if max_seconds > 0:
      speech_array = speech_array[:max_seconds*self.default_sampling_rate]
    return sf.write(save_temp_path, speech_array.numpy(), samplerate=self.default_sampling_rate)

  # Get audio duration
  def get_duration(self, audio_path):
    return librosa.get_duration(filename=audio_path)

  # clear background noise of audio by spleeter.
  def clear_background_noise(self, audio_name, current_audio_path, clear_path='clear/'):
    '''
      The function to clear background noise by Spleeter
    '''
    # Run spleeter
    subprocess.run(["spleeter", "separate", "-o", clear_path, current_audio_path])
    # copy Vocal file
    if current_audio_path=="temp_audio_cut.wav":
      self.sampling_rate_converter(clear_path+f'temp_audio_cut/vocals.wav',
                                  clear_path+'temp.wav')
    else:
      self.sampling_rate_converter(clear_path+f'{audio_name}/vocals.wav',
                                  clear_path+'temp.wav')
    # Delete spleeter folder 
    if os.path.exists(clear_path+f'{audio_name}'):
      shutil.rmtree(clear_path+f'{audio_name}')
    return clear_path+'temp.wav'
    
  def blank_audio(self, duration, channel=2, sampwidth=2, save_path="blank.wav"):
    '''
      Create blank audio
    '''
    zero_array = np.zeros(int(duration*self.default_sampling_rate), dtype = np.float64)
    audio = np.array([zero_array]*channel).T.astype("<h").tobytes()
    with wave.open("blank.wav", "w") as f:
      f.setnchannels(channel)
      f.setsampwidth(sampwidth)
      f.setframerate(self.default_sampling_rate)
      f.writeframes(audio)
    return save_path
    
  def concatenate_audio(self, first_audio_path, second_audio_path, return_path, blank_duration=0.3):
    blank_audio_path = self.blank_audio(blank_duration)
    infiles=[first_audio_path, blank_audio_path, second_audio_path]
    wave_list=[]
    for infile in infiles:
        w = wave.open(infile, 'rb')
        wave_list.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
    output = wave.open(return_path, 'wb')
    output.setparams(wave_list[0][0])
    for i in range(len(wave_list)):
        output.writeframes(wave_list[i][1])
    output.close()
      
  def get_time(self, text):
    pattern = r'\[\s*(\d{2}:\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d+)\]'
    match = re.search(pattern, text)
    if match:
      start_time = match.group(1)
      end_time = match.group(2)
    return self.to_seconds(start_time), self.to_seconds(end_time)

  def audio_segmentation(self, 
                         audio_path):
    '''
      Get audio segmentation list by Pyannote Segmentation 3.0
    '''
    return self.audio_segmentation_pipeline(audio_path)

  def get_caption(self, 
                  audio_path):
    '''
    Get audio caption by Whisper v3.
    '''
    caption=self.audio_caption_pipeline(audio_path, generate_kwargs={"language": self.language})
    return caption['text']

  def write_caption(self, 
                    audio_path, 
                    content):
    '''
      Write caption to txt file
    '''

    f=open(self.result_caption_path,"a", encoding= self.encoding)
    f.write(f"\n{audio_path}"+"|"+f"{content}")
    f.close()

  def data_generation_process(self, 
                              idx, 
                              current_audio_path, 
                              segment_start_dur=0.0,
                              skip_last_segment=0):
    '''
      The function to generate every data sample. 
    '''
    
    temp_audio_path=self.clear_background_noise(idx, current_audio_path)
    segmentation=self.audio_segmentation(temp_audio_path)
    #for turn, _, speaker in segmentation.itertracks(yield_label=True):
    # start, end = get_time(turn)
    timestamp=segmentation.to_lab().split('\n')[:-1]
    if len(timestamp)==0: return None
    # merge audio if it short than `min_merge_allow`
    previous_start_dur=-1.0
    previous_end_dur=-1.0
    for i in range(len(timestamp)-skip_last_segment):
      try:
        start, end, _ =timestamp[i].split() # Get start, end time of segment
        dur=float(end)-float(start)
        if self.min_duration>=dur: continue
        # count total duration.
        if self.return_total_duration: self.count_duration+=dur
        if dur<self.min_merge_allow:
          previous_start_dur=float(start)
          previous_end_dur=float(end)
          continue
        # save path
        return_path=self.result_audio_folder+f'/{idx}'+'['+f'{str(segment_start_dur+float(start)).replace(".", ",")}'+':'+f'{str(segment_start_dur+float(end)).replace(".", ",")}].wav'
        # audio spliting
        self.audio_spliting(temp_audio_path, float(start), float(end), return_path)
        
        if previous_start_dur>-1:
          self.audio_spliting(temp_audio_path, previous_start_dur, previous_end_dur, "short_temp.wav")
          previous_end_dur=-1
          previous_start_dur=-1
          self.concatenate_audio('short_temp.wav', return_path, return_path)
        # write caption
        audio_content=self.get_caption(return_path)
        self.write_caption(return_path, audio_content)
      except: 
        logging.critical(f"There is an unexpected error oscur\t--> Skip current segment")
        continue
    if skip_last_segment:
      return  timestamp[-1].split()[:2]
    return start, end


  # Main Function.
  def generate(self, 
           idx_path):
    # Setup download.
    logging.info('Start to setup...')
    try:
      self.init_download()
      if self.return_total_duration: self.count_duration=0
    except Exception as err:
      logging.error(err) 

    # Open Video Youtube ID.
    try: 
      f=open(idx_path, "r")
      idx_list=f.read().split('\n')
      if not len(idx_list): raise Exception("Found empty Id list!")
      f.close()
    except Exception as err:
      logging.error(err)
    
    logging.info(f'Got {len(idx_list)} ID.\n Start to generate data.')
    for i in tqdm(range(len(idx_list))):
      # Download Audio.
      current_audio_path=self.video_download(idx_list[i])
      if not current_audio_path:
        logging.critical(f"Found invalid Youtube ID:\t{idx_list[i]}\t--> Skip")
        continue

      # Convert Audio Sampling rate.
      #if librosa.get_samplerate(current_audio_path)!=self.default_sampling_rate: 
        #self.sampling_rate_converter(current_audio_path, current_audio_path)
      # Get Audio Durration
      current_audio_duration=self.get_duration(current_audio_path)
      # Case 1. Audio duration <= 600s
      if current_audio_duration<600:
        # Maximum Duration for this process: 10 minutes
        self.data_generation_process(idx_list[i], current_audio_path)
      # Case 2. Audio duration > 600s
      else:
        duration_count=0.0
        while duration_count<=current_audio_duration:
          dur_cut = duration_count+600.0 # Check durration.
          # check final cutting-portion of audio.
          if dur_cut>current_audio_duration: 
            dur_cut=current_audio_duration

          # Temprary audio file with limit at 600s
          self.audio_spliting(current_audio_path, duration_count, dur_cut, "temp_audio_cut.wav")
          
          # Case 1: Not final portion.
          if dur_cut!=current_audio_duration:
            start_dur, end_dur=self.data_generation_process(idx_list[i], "temp_audio_cut.wav", duration_count, 1)
            duration_count+=float(start_dur)
          # Case 2: Final portion.
          else: 
            self.data_generation_process(idx_list[i], "temp_audio_cut.wav", duration_count)
            break

    logging.info(f'Got {len(os.listdir(self.result_audio_folder))} samples'+'.' if not self.return_total_duration else f' with {self.count_duration/60/60} hours.')
