import argparse
from audio_data_crawling import audio_auto_crawling
def main():
  parser = argparse.ArgumentParser(description='Youtube Data Crawling')
  parser.add_argument('--huggingface_token', type=str, default="", help='Huggingface_token')
  parser.add_argument('--source_website', type=str, default="https://www.youtube.com/watch?v={}", help='Audio Source Website')
  parser.add_argument('--raw_audio_save_path', type=str, default="/raws/", help='Where to save raw audio path')
  parser.add_argument('--result_caption_path', type=str, default="caption.txt", help='Audio caption text file path')
  parser.add_argument('--result_audio_folder', type=str, default="./result_wav", help='Result audio folder path')
  parser.add_argument('--skip_title', type=str, default='[]', help='Skipping audio if title contain words...')
  parser.add_argument('--reverse_skip', type=bool, default=False, help='Skip title if title not contain above word...')
  parser.add_argument('--return_total_duration', type=bool, default=False, help='Print total audio data duration')
  parser.add_argument('--language', type=str, default="vietnamese", help='Language for ASR model')
  parser.add_argument('--IDpath', type=str, default="./", help='text file contain audio ID')
  parser.add_argument('--ffmpeg_location', type=str, default="/usr/bin/ffmpeg", help='ffmpeg for yt_dlp')
  parser.add_argument('--use_spleeter', type=bool, default=True, help='Using Spleeter to remove music background')
  parser.add_argument('--use_MVSEP', type=bool, default=False, help='Using MVSEP to remove music background')
  parser.add_argument('--use_deepfiller3', type=bool, default=False, help='Using deepfiller3 to enhance audio quality')
  parser.add_argument('--taskID', type=int, default=0, help='Data for T2S: 0 | ASR: 1')
  args = parser.parse_args()

  run=audio_auto_crawling(               
               args.huggingface_token, 
               args.source_website, 
               args.raw_audio_save_path,
               args.result_caption_path,
               args.result_audio_folder,
               list(eval(args.skip_title)),
               args.reverse_skip,
               args.return_total_duration,
               args.language,
               args.use_spleeter,
               args.use_MVSEP,
               args.use_deepfiller3,
               args.taskID
              )
  
  run.generate(args.IDpath)

if __name__ == "__main__":
    main()
