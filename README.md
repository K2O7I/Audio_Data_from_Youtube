# Audio_Data_from_Youtube
***
#### Author: Nguyen Minh Quan.
-This program is a part of DataStudio Project.-
#### Version 1.0.0
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
   `git clone https://github.com/K2O7I/ASR_Data_from_Youtube.git`
   - Step 2:\
   `pip install -r requirements.txt`  
   - Step 3:\
    Get your *[Huggingface tokens](https://huggingface.co/settings/tokens)*.
   - Step 4:\
    Accept the [Pyannote policy](https://huggingface.co/pyannote/segmentation-3.0) with your Huggingface account.
   - Step 5:\
    `python3 main.py ----huggingface_token <YOUR_HF_TOKEN> --IDpath <./videoID.txt>`
   - Step 6:\
     If you keep the default setup, the result of this program will be found in 2 locations:
     * The **result_wav** folder.
     * The **caption.txt** text file. <br>--> This text file store video caption by this template: *<VIDEO_NAME>|<VIDEO_CAPTION>*.
4. **Future update.**
    - [ ] Multi-threaded running.
    - [ ] Option to choose other AI models.
    - [ ] Automatically correct caption spelling.
    - [ ] Boots up speed.
    - [ ] Auto push result to Huggingface dataset.
5. **Reference.**
   * OpenAI/[Whisper-Large-v3](https://huggingface.co/openai/whisper-large-v3).
   * Pyannote/[Segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0).
