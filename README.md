# Audio_Data_from_Youtube
***
#### Author: Nguyen Minh Quan.
-This program is a part of DataStudio Project.-
#### Version 1.0.1
---
1. **About.**\
   This is the program used to download audio data from YouTube and label it.\
   The data from this program could be applied for Automatic Speech Recognition (ASR) and Text-to-speech (T2S) tasks.\
   This program collects YouTube audio by its ID. The <videoID.txt> has the following structure:\
     <br>
     ID0001<br>
     ID0002<br>
     ...<br>
   This program currently utilizes OpenAI Whisper-v3 and Pyannote-segmentation-3.0.\
   *The audio from YouTube may contain **copyright**. The author of this code will not be responsible for any violation of the data collected from this program.*
3. **How to run.**
   - Step 1:\
   if using conda:\
      &nbsp;`conda install ffmpeg` \
   else: \
      &nbsp;`apt install ffmpeg`\
     --> Using `whereis ffmpeg` to find *location of ffmpeg*.
   - Step 2:\
   &nbsp;`git clone https://github.com/K2O7I/Audio_Data_from_Youtube.git`
   - Step 3: \
   &nbsp;`cd Audio_Data_from_Youtube`
   - Step 4:\
   &nbsp;`pip install -r requirements.txt`<br>`pip install tensorflow==2.15.0`  
   - Step 5:\
   &nbsp;Get your *[Huggingface tokens](https://huggingface.co/settings/tokens)*.
   - Step 6:\
   &nbsp;Accept the [Pyannote policy](https://huggingface.co/pyannote/segmentation-3.0) with your Huggingface account.
   - Step 7:\
    &nbsp;`python3 main.py ----huggingface_token <YOUR_HF_TOKEN> --IDpath <./videoID.txt> --ffmpeg_location "<ffmpeg_Location>"`
   - Step 8:\
     &nbsp;If you keep the default setup, the result of this program will be found in 2 locations:
     &nbsp;&nbsp;* The **result_wav** folder.
     &nbsp;&nbsp;* The **caption.txt** text file. <br>--> This text file store video caption by this template: *<VIDEO_NAME>|<VIDEO_CAPTION>*.
   <br>
   
   **Sample command in Google Colab:** 
   &nbsp;```
   python3 main.py --huggingface_token "hf_1234567890" \
                 --IDpath "../ids.txt" \
                 --skip_title "['[Music]', '[Podcast]']" \
                 --reverse_skip True \
                 --raw_audio_save_path "/content/raw" \
                 --result_audio_folder "/content/result_audio_folder" \
                 --result_caption_path "/content/caption.txt"
   ```
5. **Future update.**
    - [ ] Multi-threaded running.
    - [ ] Option to choose other AI models.
    - [ ] Automatically correct caption spelling.
    - [ ] Boots up speed.
    - [ ] Auto push result to Huggingface dataset.
6. **Reference.**
   * OpenAI/[Whisper-Large-v3](https://huggingface.co/openai/whisper-large-v3).
   * Pyannote/[Segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0).
