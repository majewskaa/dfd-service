class NoFaceDetectedError(Exception):
    def __init__(self, message: str = "No face detected in the video."):
        Exception.__init__(self, message)