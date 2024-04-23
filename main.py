import argparse
from asr_data_crawling import asr_auto_crawling
def main():
  parser = argparse.ArgumentParser(description='Youtube Data Crawling')
  parser.add_argument('--huggingface_token', type=str, default="", help='Huggingface_token')
  parser.add_argument('--source_website', type=str, default="https://www.youtube.com/watch?v={}", help='Huggingface_token')
  parser.add_argument('--raw_audio_save_path', type=str, default="/raws/", help='Huggingface_token')
  parser.add_argument('--result_caption_path', type=str, default="caption.txt", help='Huggingface_token')
  parser.add_argument('--result_audio_folder', type=str, default="./result_wav", help='Huggingface_token')
  parser.add_argument('--skip_title', type=str, default='[]', help='Huggingface_token')
  parser.add_argument('--reverse_skip', type=bool, default=False, help='Huggingface_token')
  parser.add_argument('--return_total_duration', type=bool, default=False, help='Huggingface_token')
  parser.add_argument('--language', type=str, default="vietnamese", help='Huggingface_token')
  parser.add_argument('--IDpath', type=str, default="./", help='Huggingface_token')
  args = parser.parse_args()

  run=asr_auto_crawling(               
               args.huggingface_token, 
               args.source_website, 
               args.raw_audio_save_path,
               args.result_caption_path,
               args.result_audio_folder,
               list(args.skip_title),
               args.reverse_skip,
               args.return_total_duration,
               args.language)
  
  run.generate(args.IDpath)

if __name__ == "__main__":
    main()