from src.toolsjasinski._parse_video import parse_video

path = r'src\toolsjasinski\tests\Count00000_Channelred,green_Seq0000.nd2'

video = parse_video(path)
# print(video.attributes)
# print('\n',video.frame_metadata(1))
# print('\n',video.experiment)
# print('\n',video.text_info)
print(video)