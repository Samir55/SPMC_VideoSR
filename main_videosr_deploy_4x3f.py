import os
import time
import glob
import numpy as np
import tensorflow as tf
import subprocess
from datetime import datetime
from math import ceil
import scipy.misc
from video_handler import VideoHandler

from modules.videosr_ops_lite import *

os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
    "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

DATA_TEST = './data/test/car05_001'
DATA_TRAIN = './data/train/'


class VIDEOSR(object):

    def test(self, scale_factor=4, num_frames=3):
        data_path = DATA_TEST
        in_list = sorted(glob.glob(os.path.join(data_path, 'input{}/*.png').format(scale_factor)))

        # Read images and crop the top left 120 * 160 pixels and normalize them.
        # inp = [scipy.misc.imresize(i, [120, 160]) / 255.0 for i in inp]
        inp = [scipy.misc.imread(i).astype(np.float32) / 255.0 for i in in_list]
        inp = [i[:120, :160, :] for i in inp]

        print('Testing path: {}'.format(data_path))
        print('# of testing frames: {}'.format(len(in_list)))

        # Create the output folder.
        data_test_out = DATA_TEST + '/../output/_SR_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(data_test_out)

        cnt = 0
        for idx0 in range(len(in_list)):
            cnt += 1
            t = int(num_frames / 2)

            images = [inp[0] for i in range(idx0 - t, 0)]
            images.extend([inp[i] for i in range(max(0, idx0 - t), idx0)])
            images.extend([inp[i] for i in range(idx0, min(len(in_list), idx0 + t + 1))])
            images.extend([inp[-1] for i in range(idx0 + t, len(in_list) - 1, -1)])

            # Pre processing and padding.
            dims = images[0].shape
            if len(dims) == 2:
                images = [np.expand_dims(i, -1) for i in images]
            h, w, c = images[0].shape
            out_h = h * scale_factor
            out_w = w * scale_factor
            pad_h = int(ceil(h / 4.0) * 4.0 - h)
            pad_w = int(ceil(w / 4.0) * 4.0 - w)
            images = [np.pad(i, [[0, pad_h], [0, pad_w], [0, 0]], 'edge') for i in images]
            images = np.expand_dims(np.stack(images, axis=0), 0)

            # Initialize the model placeholders and load saved model at the first iteration only.
            if idx0 == 0:
                frames_lr = tf.placeholder(dtype=tf.float32, shape=images.shape)
                frames_ref_ycbcr = rgb2ycbcr(frames_lr[:, t:t + 1, :, :, :])
                frames_ref_ycbcr = tf.tile(frames_ref_ycbcr, [1, num_frames, 1, 1, 1])

                with open('spmc_120_160_4x3f.pb', 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    output = tf.import_graph_def(graph_def, input_map={'Placeholder:0': frames_lr},
                                                 return_elements=['output:0'])
                    output = output[0]
                    print(output.get_shape())

                if len(dims) == 3:
                    output_rgb = ycbcr2rgb(tf.concat([output, resize_images(frames_ref_ycbcr,
                                                                            [(h + pad_h) * scale_factor,
                                                                             (w + pad_w) * scale_factor],
                                                                            method=2)[:, :, :, :, 1:3]], -1))
                else:
                    output_rgb = output
                output = output[:, :, :out_h, :out_w, :]
                output_rgb = output_rgb[:, :, :out_h, :out_w, :]

            if cnt == 1:
                sess = tf.Session()

            case_path = data_path.split('/')[-1]
            print('Testing - ', case_path, len(images))

            # Feed forward the model and get the output.
            frame_start = time.clock()
            print(frames_lr)
            [images_hr, images_hr_rgb] = sess.run([output, output_rgb], feed_dict={frames_lr: images})
            elapsed = time.clock() - frame_start
            print("Single frame convertion elapsed time: ", elapsed*1000, " ms")

            # Save the frame.
            scipy.misc.imsave(os.path.join(data_test_out, 'y_%03d.png' % (idx0)),
                              im2uint8(images_hr[0, -1, :, :, 0]))
            if len(dims) == 3:
                scipy.misc.imsave(os.path.join(data_test_out, 'rgb_%03d.png' % (idx0)),
                                  im2uint8(images_hr_rgb[0, -1, :, :, :]))
        print('SR results path: {}'.format(data_test_out))


def main(_):
    model = VIDEOSR()
    model.test()


if __name__ == '__main__':
    tf.app.run()
