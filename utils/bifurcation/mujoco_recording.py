import shutil
import subprocess

import mujoco


class MP4Recorder:
    """Small MuJoCo offscreen recorder with optional MP4 backends."""

    def __init__(self, model, path, fps=60, width=1280, height=720):
        self.model = model
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.renderer = None
        self.backend = None
        self.writer = None
        self.process = None

        try:
            import imageio.v2 as imageio

            self.backend = "imageio"
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.writer = imageio.get_writer(path, fps=fps, codec="libx264")
            return
        except ImportError:
            pass

        try:
            import cv2

            self.backend = "cv2"
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            return
        except ImportError:
            pass

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            self.backend = "ffmpeg"
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.process = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{width}x{height}",
                    "-pix_fmt",
                    "rgb24",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        raise RuntimeError(
            "MP4 recording needs one encoder backend: install imageio[ffmpeg], "
            "opencv-python, or system ffmpeg."
        )

    def capture(self, data):
        self.renderer.update_scene(data)
        frame = self.renderer.render()

        if self.backend == "imageio":
            self.writer.append_data(frame)
        elif self.backend == "cv2":
            import cv2

            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif self.backend == "ffmpeg":
            self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.backend == "imageio":
            self.writer.close()
        elif self.backend == "cv2":
            self.writer.release()
        elif self.backend == "ffmpeg":
            self.process.stdin.close()
            self.process.wait()
        self.renderer.close()
