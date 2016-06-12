# -*- coding: utf-8 -*-

import os
import time
import pycurl
from cStringIO import StringIO

import cv2
import numpy
from PIL import Image


class FeatureBasedImageMatch(object):
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH = 6

    """ 基于图片特征的匹配算法 """
    def __init__(self, detector='SURF', matcher='FLANN'):
        """
        :param detector: string SURF/SIFT/ORB/BRISK 计算特征所使用的算法,SURF比SIFT更快,但损失部分准确性
        :param matcher: string FLANN/BF 特征匹配所使用的算法,BF为全面检测,FLANN为相邻匹配,数据量大的情况下FLANN更快
        Description:
        生成特征算法参考
        1.http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
        2.http://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html
        匹配算法选择参考
        1.http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html#BFMatcher%20:%20public%20DescriptorMatcher
        """
        # 计算特征算法选择
        if detector == 'SURF':
            self.detector = cv2.SURF(500, nOctaves=4, nOctaveLayers=2, extended=0, upright=1)
            norm = cv2.NORM_L2
        elif detector == 'SIFT':
            self.detector = cv2.SIFT()
            norm = cv2.NORM_L2
        elif detector == 'ORB':
            self.detector = cv2.ORB(400)
            norm = cv2.NORM_HAMMING
        elif detector == 'BRISK':
            self.detector = cv2.BRISK()
            norm = cv2.NORM_HAMMING
        else:
            raise FeatureBasedImageMatchException('Detector %s not support yet.' % detector)
        # 特征匹配算法选择
        if matcher == 'FLANN':
            if norm == cv2.NORM_L2:
                flann_params = dict(algorithm=self.__class__.FLANN_INDEX_KDTREE, trees=5)
            else:
                flann_params = dict(algorithm=self.__class__.FLANN_INDEX_LSH,
                                    table_number=6,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=1)  # 2
            self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        elif matcher == 'BF':
            self.matcher = cv2.BFMatcher(norm)
        else:
            raise FeatureBasedImageMatchException('Matcher %s not support yet.' % matcher)

        self.train_descriptors = []

    def detect_and_compute(self, image=None, image_file=None):
        """ 探测及计算特征 """
        if not isinstance(image, numpy.ndarray) and not image_file:
            raise FeatureBasedImageMatchException('Param image or image_file must needed.')

        if not isinstance(image, numpy.ndarray):
            image = ImageMatch.image_read(image_file)

        key_points, descriptors = self.detector.detectAndCompute(image, None)
        return key_points, descriptors

    def many_image_match(self, descriptors, ratio=0.75):
        """ 与匹配器中的特征匹配获取最高匹配特征值 """
        matches = self.matcher.knnMatch(descriptors, k=2)
        return self.__class__.filter_matches(matches, ratio)

    def image_match(self, descriptors_1, descriptors_2, ratio=0.75):
        """ 与匹配器中的特征匹配获取最高匹配特征值 """
        try:
            matches = self.matcher.knnMatch(descriptors_1, trainDescriptors=descriptors_2, k=2)
        except TypeError as e:
            print (e)
            return []
        except cv2.error as e:
            print (e)
            return []

        return self.__class__.filter_matches(matches, ratio)

    def image_match_explore(self, key_points_1, descriptors_1, key_points_2, descriptors_2, ratio=0.75):
        """ 单独两张图的特征匹配 """
        matches = self.matcher.knnMatch(descriptors_1, trainDescriptors=descriptors_2, k=2)
        points_1, points_2, matches = ImageMatch.filter_matches(key_points_1, key_points_2, matches, ratio)
        return points_1, points_2, matches

    def add_matches_image(self, descriptors):
        """ 将图片特征添加到匹配器中 """
        try:
            self.matcher.add([descriptors])
        except TypeError:
            return False

        return True

    def clear_matches_image(self):
        """ 清空匹配器中的图片特征 """
        self.matcher.clear()

    @staticmethod
    def filter_matches(matches, ratio=0.75):
        """ 筛选有效特征 """
        return [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]


class FeatureBasedImageMatchException(Exception):
    pass


class ImageMatch(object):
    """ 图片匹配的工具类 """
    def __init__(self):
        self.file_dir = '/tmp/'
        self.file_ext = '.npy'

    @staticmethod
    def image_read(image_file=None, image_buffer=None):
        """ 载入图片 """
        # 载入灰度图,OpenCV的图片读取只能使用本地文件路径,换成PIL读取,最终结果会有细微差别,忽略
        if image_buffer:
            im = Image.open(StringIO(image_buffer))
        else:
            if not os.path.isfile(image_file):
                c = pycurl.Curl()
                c.setopt(pycurl.URL, image_file)
                buf = StringIO()
                c.setopt(pycurl.WRITEFUNCTION, buf.write)
                c.setopt(pycurl.TIMEOUT, 5)
                try:
                    c.perform()
                except pycurl.error as e:
                    print e
                    return False
                image_buffer = buf.getvalue()
                buf.close()
                c.close()
                if not image_buffer:
                    return False
                im = Image.open(StringIO(image_buffer))

            else:
                im = Image.open(image_file)

        image = numpy.asarray(im.convert('L'))

        return image

    @staticmethod
    def filter_matches(kp1, kp2, matches, ratio=0.75):
        """ 筛选匹配特征 """
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
        p1 = numpy.float32([kp.pt for kp in mkp1])
        p2 = numpy.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, list(kp_pairs)

    @staticmethod
    def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
        """ 预览匹配到的特征 """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = numpy.zeros((max(h1, h2), w1 + w2), numpy.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = numpy.ones(len(kp_pairs), numpy.bool_)
        p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
        for kpp in kp_pairs:
            p1.append(numpy.int32(kpp[0].pt))
            p2.append(numpy.int32(numpy.array(kpp[1].pt) + [w1, 0]))

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
                cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
                cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
                cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
        vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)

        cv2.imshow(win, vis)
        cv2.waitKey()
        cv2.destroyWindow(win)
        return vis

    def feature_save(self, picture_md5, feature):
        """ 保存特征 """
        filename = self.get_feature_path(picture_md5)
        return numpy.save(filename, feature)

    def feature_load(self, picture_md5):
        """ 加载特征 """
        filename = self.get_feature_path(picture_md5)
        if not filename.endswith('.npy'):
            filename += '.npy'
        return numpy.load(filename)

    def get_feature_path(self, picture_md5):
        """ 获取特征所在文件 """
        return self.file_dir + picture_md5 + self.file_ext

    def feature_exist(self, picture_md5):
        """ 检测特征文件是否存在 """
        filename = self.get_feature_path(picture_md5)
        if not filename.endswith('.npy'):
            filename += '.npy'
        return os.path.isfile(filename)

    def feature_remove(self, picture_md5):
        """ 删除特征文件 """
        filename = self.get_feature_path(picture_md5)
        if not filename.endswith('.npy'):
            filename += '.npy'
        return os.remove(filename)


class timer:
    """ 耗时计时器 """
    def __init__(self, func=time.time):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

if __name__ == '__main__':
    match_object = FeatureBasedImageMatch()
    # 样本图片
    images = [
        'http://image-qzone.mamaquan.mama.cn/upload/2016/05/03/thumb_w196_bf6022f770f3b5f1eecc_w300X400_w192X256.jpg',
    ]
    # 待检测图片
    match_image = 'http://image-qzone.mamaquan.mama.cn/upload/2016/05/03/thumb_w196_4ab517ac123cc9abffbe_w300X400_w192X256.jpg'
    match_image_object = ImageMatch.image_read(match_image)
    match_key_point, match_descriptors = match_object.detect_and_compute(image=match_image_object)
    image_object = dict()

    # 单独两张图的特征匹配
    x = images[0]
    image_object[x] = dict()
    image_object[x]['image'] = ImageMatch.image_read(x)
    with timer() as t:
        image_object[x]['key_points'], image_object[x]['descriptors'] = match_object.detect_and_compute(
            image=image_object[x]['image'])
    print('Finish detect and compute image feature:%d and cost%f' % (len(image_object[x]['key_points']), t.elapsed))
    with timer() as t:
        points_1, points_2, matches = match_object.image_match_explore(match_key_point, match_descriptors,
                                                                       image_object[x]['key_points'],
                                                                       image_object[x]['descriptors'])
    print('Finish image match,match image feature:%d and cost%f' % (len(matches), t.elapsed))
    ImageMatch.explore_match('feature_match', match_image_object, image_object[x]['image'], matches)

    # 多张图片的特征匹配最大值
    for x in images:
        image_object[x] = dict()
        image_object[x]['image'] = ImageMatch.image_read(x)
        with timer() as t:
            image_object[x]['key_points'], image_object[x]['descriptors'] = match_object.detect_and_compute(image=image_object[x]['image'])
        print('Finish detect and compute image feature:%d and cost%f' % (len(image_object[x]['key_points']), t.elapsed))
        with timer() as t:
            match_object.add_matches_image(image_object[x]['descriptors'])
        print('Finish add matches image and cost%f' % t.elapsed)

    with timer() as t:
        matches = match_object.many_image_match(match_descriptors)
    print('Finish image match,match image feature:%d and cost%f' % (len(matches), t.elapsed))
