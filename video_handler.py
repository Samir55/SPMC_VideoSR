import skvideo.io
import skvideo.datasets


# This class represents the video class handler, it's responsible for getting the frames needed for feature extraction.
class VideoHandler:
    # Get a video reader to read frame by frame, we don't load all the frames into the memory.
    @staticmethod
    def get_frame_reader(video_path):
        return skvideo.io.vreader(video_path)
