import cv2
import numpy as np

def fetch_chunk_frames(cap, n_chunk_frames, step_frame, size=None):
    frames = []
    cap_type = 'cv2_cap' if isinstance(cap, cv2.VideoCapture) else 'yuv_cap'
    for f in range(n_chunk_frames * step_frame):
        assert cap.isOpened()
        if f % step_frame == 0:
            ret, frame = cap.read()
        else:
            ret, _ = cap.read_raw() if cap_type == 'yuv_cap' else cap.read()

        if not ret:
            break
        if f % step_frame == 0:
            print(f)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if size is not None:
                frame = cv2.resize(frame, size)
            frames.append(frame)
    return frames

def get_config(args):
    start_time = args.start_time
    end_time = args.end_time

    hr_path = args.hr_video
    lr_path = args.lr_video

    hr_size = args.hr_size.split(',')
    hr_size = (int(hr_size[0]), int(hr_size[1]))
    if args.lr_size is not None:
        lr_size = args.lr_size.split(',')
        lr_size = (int(lr_size[0]), int(lr_size[1]))
    else:
        lr_size = (hr_size[0] // 4, hr_size[1] // 4)

    hr_vid_format = hr_path.split('.')[-1]
    lr_vid_format = lr_path.split('.')[-1]

    hr_cap = VideoCaptureYUV(hr_path, hr_size) if hr_vid_format == 'yuv' else cv2.VideoCapture(hr_path)
    if hr_vid_format == 'yuv':
        fps = args.fps
        if len(fps.split('/')) == 2:
            fps = float(fps.split('/')[0]) / float(fps.split('/')[1])
            fps = int(round(fps))
        else:
            fps = int(round(float(fps)))
    else:
        fps = int(round(hr_cap.get(cv2.CAP_PROP_FPS)))
    hr_cap.release()

    step_time = args.sampling_interval
    step_frame = max(1, int(step_time * fps))
    n_chunk_frames = int(args.update_interval * fps / float(step_frame))
    boundary_threshold = ((n_chunk_frames * args.num_epochs) // args.batch_size) * args.batch_size
    if args.inference:
        boundary_threshold = n_chunk_frames

    total_chunks = int((args.end_time - args.start_time) / args.update_interval)
    warmup_epochs = 0 if args.coord_frac == 1.0 else 1
    num_epochs = args.num_epochs - warmup_epochs
    n_chunk_iterations = (n_chunk_frames * num_epochs) // args.batch_size
    num_frames = total_chunks * n_chunk_frames

    config = {'n_chunk_frames': n_chunk_frames,
              'hr_path': hr_path,
              'hr_size': hr_size,
              'lr_path': lr_path,
              'lr_size': lr_size,
              'fps': fps,
              'boundary_threshold': boundary_threshold,
              'step_frame': step_frame,
              'hr_vid_format': hr_vid_format,
              'lr_vid_format': lr_vid_format,
              'warmup_epochs': warmup_epochs,
              'num_epochs': num_epochs,
              'total_chunks': total_chunks,
              'n_chunk_iterations': n_chunk_iterations,
              'num_frames': num_frames,
              'start_time': start_time,
              'end_time': end_time, }
    return config


m = np.array([[1.0, 1.0, 1.0],
              [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
              [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])
m = m[..., [2, 1, 0]]


def YUV2BGR(yuv):
    h = int(yuv.shape[0] / 1.5)
    w = yuv.shape[1]
    y = yuv[:h]
    h_u = h // 4
    h_v = h // 4
    u = yuv[h:h + h_u]
    v = yuv[-h_v:]
    u = np.reshape(u, (h_u * 2, w // 2))
    v = np.reshape(v, (h_v * 2, w // 2))
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_CUBIC)
    yuv = np.concatenate([y[..., None], u[..., None], v[..., None]], axis=-1)

    bgr = np.dot(yuv, m)
    bgr[:, :, 2] -= 179.45477266423404
    bgr[:, :, 1] += 135.45870971679688
    bgr[:, :, 0] -= 226.8183044444304
    bgr = np.clip(bgr, 0, 255)

    return bgr.astype(np.uint8)


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.width, self.height = size
        self.frame_len = (self.width * self.height * 3) // 2
        self.f = open(filename, 'rb')
        self.is_opened = True
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            self.is_opened = False
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, None
        bgr = YUV2BGR(yuv)
        return ret, bgr

    def isOpened(self):
        return self.is_opened

    def release(self):
        try:
            self.f.close()
        except Exception as e:
            print(str(e))
