# coding:utf-8
import cv2
import numpy as np
import os
import sys
import copy
from scipy.spatial.distance import cityblock
from tqdm import tqdm
'''
基于直方图的帧间差法
'''


class shot_detector:
    def __init__(self,
                 video_path=None,
                 min_duration=10,
                 output_dir=None,
                 thres=1.5):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = output_dir
        self.hist_size = 64  # how many bins for each R,G,B histogram

        # any transition must be no less than this threshold range from 0 to 3, the higher the more sensitive.
        self.absolute_threshold = thres

    def get_normed_hist(self, frame):
        '''
        计算直方图并归一化、扁平化(展开3个通道)
        @return 1维数组, 元素个数hist_size*通道数
        '''
        color_hist = [cv2.calcHist([frame], [c], None, [self.hist_size], [0.0, 255.0])
                      for c in range(3)]
        color_hist = np.array(
            [hist_c / float(sum(hist_c)) for hist_c in color_hist])
        return color_hist.flatten()  # 将3个通道展开成1维

    def run_detect(self, video_path=None, batch_size=100):
        '''
        read frames into memory part by part
        '''
        if video_path is not None:
            self.video_path = video_path
        assert (self.video_path is not None), "video_path is None."
        assert (os.path.exists(self.video_path)), "video path is invalid."

        threshold = self.absolute_threshold  # 阈值
        cap = cv2.VideoCapture(self.video_path)

        IMG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 视频帧宽
        IMG_HEIGHT= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 视频帧高
        if IMG_WIDTH > IMG_HEIGHT:
            IMG_SIZE = (256, int(256.0 * float(IMG_HEIGHT) / float(IMG_WIDTH)))
        else:
            IMG_SIZE = (int(256.0 * float(IMG_WIDTH) / float(IMG_HEIGHT)), 256)
        print('IMG_SIZE: ', IMG_SIZE)
        FRAME_NUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频所有帧数

        print('--processing video...')
        key_frames = []
        for i in tqdm(range(1, FRAME_NUM)):
            success, frame = cap.read()
            if not success:  # 判断当前帧是否存在
                break

            # 视频帧预处理
            frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_CUBIC)

            if i == 1:  # 记录往前2帧
                frame_prev_2 = frame
                continue
            elif i == 2:  # 记录往前1帧
                frame_prev_1 = frame
                continue
            else:  # 从第3帧开始判断
                hist_prev_1 = self.get_normed_hist(frame_prev_1)
                hist_prev_2 = self.get_normed_hist(frame_prev_2)
                hist_cur = self.get_normed_hist(frame)

                # 计算两个histgram的曼哈顿距离
                score_pre = cityblock(hist_prev_1, hist_prev_2)
                score_cur = cityblock(hist_cur, hist_prev_1)

                # 一阶,二阶差分阈值判断
                if (score_cur >= threshold) \
                        and (abs(score_cur - score_pre) >= threshold * 0.5):
                    key_frames.append(frame)  # 记录关键帧

                # 更新前两帧
                frame_prev_2 = frame_prev_1  # 更新往前1帧
                frame_prev_1 = frame  # 更新往前2帧

        # 处理长镜头(整个视频没有检测到镜头切换)
        if len(key_frames) == 0:
            key_frames.append(frame_prev_2)
            key_frames.append(frame_prev_1)
        cap.release()  # 释放资源
        return key_frames, IMG_SIZE

    def run(self, video_path=None):
        '''
        此接口适合处理短视频, 可以将全部帧数据加载到内存中
        '''
        if video_path is not None:
            self.video_path = video_path
        assert (self.video_path is not None), "video_path is None."
        assert (os.path.exists(self.video_path)), "video path is invalid."
        self.shots = []  # 镜头初始化为空
        hists, frames = [], []

        cap = cv2.VideoCapture(self.video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break
            # if self.output_dir is not None:
            frames.append(frame)

            # compute RGB histogram for each frame
            color_hist = [cv2.calcHist([frame], [c], None, [self.hist_size], [0.0, 255.0])
                          for c in range(3)]
            color_hist = np.array(
                [hist_c / float(sum(hist_c)) for hist_c in color_hist])
            hists.append(color_hist.flatten())  # 将3个通道展开成1维

        # manhattan distance of two consecutive histgrams
        scores = [cityblock(*h_diff)
                  for h_diff in zip(hists[:-1], hists[1:])]  # cityblock: 曼哈顿距离
        print("max diff:", max(scores), "min diff:", min(scores))

        # compute automatic threshold
        # mean_score = np.mean(scores)
        # std_score = np.std(scores)
        threshold = self.absolute_threshold

        # decide shot boundaries
        prev_i = 0
        prev_score = scores[0]
        for i, score in enumerate(scores[1:]):  # 计算一阶差分与二阶差分
            if (score >= threshold) and (abs(score - prev_score) >= threshold * 0.5):
                self.shots.append((prev_i, i + 2))  # 记录镜头
                prev_i = i + 2
            prev_score = score
        video_length = len(hists)
        self.shots.append((prev_i, video_length))  # 记录最后一组镜头
        assert video_length >= self.min_duration, "duration error"
        self.merge_short_shots()

        # save key frames
        # if self.output_dir is not None:
        #     if not os.path.exists(self.output_dir):
        #         os.makedirs(self.output_dir)
        #     del_files(self.output_dir)
        #     for shot in self.shots:
        #         cv2.imwrite("%s/frame-%d.jpg" %
        #                     (self.output_dir, shot[0]), frames[shot[0]])
        #     print("key frames written to %s" % self.output_dir)

        if len(frames) != 0:
            # return key frames
            return [frames[shot[0]] for shot in self.shots]
        else:
            return None

    def merge_short_shots(self):
        # merge short shots
        while True:
            durations = [shot[1] - shot[0] for shot in self.shots]  # 镜头间隔帧数
            shortest = min(durations)  # 最短镜头间隔帧数

            # no need to merge
            if shortest >= self.min_duration:  # 如果最短的镜头间隔帧数比指定阈值大就不用合并
                break

            idx = durations.index(shortest)
            left_half = self.shots[:idx]
            right_half = self.shots[idx + 1:]
            shot = self.shots[idx]

            # can only merge left
            if idx == len(self.shots) - 1:
                left = True

            # can only merge right
            elif idx == 0:
                left = False
            else:
                # otherwise merge the shorter one
                if durations[idx - 1] < durations[idx + 1]:
                    left = True
                else:
                    left = False
            if left:
                self.shots = left_half[:-1] + \
                    [(left_half[-1][0], shot[1])] + right_half
            else:
                self.shots = left_half + \
                    [(shot[0], right_half[0][1])] + right_half[1:]


def get_key_frames(video_path):
    '''
    extract key frames from a given video
    '''
    detector = shot_detector(video_path, output_dir=None, thres=1.5)
    return detector.run_detect()


def test(in_dir, out_dir):
    detector = shot_detector(in_dir, output_dir=out_dir, thres=1.5)
    detector.run()
    print(detector.shots)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ./shotDetect.py <video-path> <output_dir>")
        sys.exit()
    video_path = sys.argv[1]
    key_frames_dir = sys.argv[2]
    detector = shot_detector(video_path, output_dir=key_frames_dir, thres=1.5)
    detector.run()
    print(detector.shots)

    # ----------debuging and testing...
    # test('./Sensitivity.avi', './key_frames/')
