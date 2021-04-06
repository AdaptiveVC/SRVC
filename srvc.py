import argparse
import os
import pathlib
import pickle
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

import models
from utils.misc import get_metrics, SaveHelper as save_helper
from utils.video_reader import VideoCaptureYUV, fetch_chunk_frames, get_config

parser = argparse.ArgumentParser(description='Run SRVC training & inference.')
parser.add_argument('hr_video', type=str, default=None, help='Directory for the HR video')
parser.add_argument('lr_video', type=str, default=None, help='Directory for the LR video')
parser.add_argument('save_path', type=str, default=None, help='Directory to save the results')
parser.add_argument('load_path', type=str, default='', help='Directory for loading the SR model checkpoint(s)')
parser.add_argument('model_name', type=str, default=None, help='SR model variant')
parser.add_argument('lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('batch_size', type=int, default=1, help='Train batch size')
parser.add_argument('start_time', tpye=float, default=0, help='Training data start time in the video')
parser.add_argument('end_time', tpye=float, default=10, help='Training data interval end time in the video')
parser.add_argument('sampling_interval', tpye=float, default=0, help='Sampling time interval (seconds)')
parser.add_argument('update_interval', tpye=float, default=10, help='Model update interval (seconds)')
parser.add_argument('coord_frac', tpye=float, default=1.0, help='Fraction of trainable model parameters')
parser.add_argument('num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('fps', type=str, default='24',
                    help='Video framerate (only required for raw video files); it also accepts formats like 24000/1001')
parser.add_argument('ff_nchunks', type=int, default=0, help='fast-forward n chunks')
parser.add_argument('inference', type=bool, default=False, help='Run inference test first')
parser.add_argument('single_checkpoint', type=bool, default=False, help='Run the inference using only one checkpoint')
parser.add_argument('crop_augment', type=bool, default=False, help='Add random crops augmentation')
parser.add_argument('l1_loss', type=bool, default=False, help='Use L1 loss instead of L2')
parser.add_argument('dump_samples', type=bool, default=False, help='Save the inference sample frames')
parser.add_argument('hr_size', type=str, default='1920,1080', help='Comma-separated HR video size, i.e.,  width,height')
parser.add_argument('lr_size', type=str, default='480,270', help='Comma-separated LR video size, i.e., width,height')
parser.add_argument('online', type=bool, default=False,
                    help='Uses the model trained till the beginning of each chunk for inference on that chunk (only '
                         'effects the inference process)')

args = parser.parse_args()

def gen_video():
    cnfg = get_config(args)
    chunk_no = args.ff_nchunks - 1
    total_frames = args.ff_nchunks * cnfg['boundary_threshold']
    hr_cap = VideoCaptureYUV(cnfg['hr_path'], cnfg['hr_size']) if cnfg['hr_vid_format'] == 'yuv' else cv2.VideoCapture(
            cnfg['hr_path'])
    if args.lr_video is not None:
        lr_cap = VideoCaptureYUV(cnfg['lr_path'], cnfg['lr_size']) if cnfg[
                                                                          'lr_vid_format'] == 'yuv' else cv2.VideoCapture(
                cnfg['lr_path'])
    else:
        lr_cap = None

    print('----> Fast-forwarding the capture')
    for n in range(args.ff_nchunks):
        fetch_chunk_frames(hr_cap,
                           cnfg['n_chunk_frames'],
                           cnfg['step_frame'],
                           None if cnfg['hr_vid_format'] == 'yuv' else cnfg['hr_size'])
        if lr_cap is not None:
            fetch_chunk_frames(lr_cap,
                               cnfg['n_chunk_frames'],
                               cnfg['step_frame'],
                               None if cnfg['lr_vid_format'] == 'yuv' else cnfg['lr_size'])
    while True:
        if total_frames % cnfg['boundary_threshold'] == 0:
            chunk_no += 1
            chunk_start = cnfg['start_time'] + chunk_no * args.update_interval
            chunk_end = cnfg['start_time'] + (chunk_no + 1) * args.update_interval
            chunk_end = min(chunk_end, cnfg['end_time'])
            if chunk_start >= cnfg['end_time'] or not hr_cap.isOpened():
                break
            chunk_frames = fetch_chunk_frames(hr_cap,
                                              cnfg['n_chunk_frames'],
                                              cnfg['step_frame'],
                                              None if cnfg['hr_vid_format'] == 'yuv' else cnfg['hr_size'])
            if lr_cap is not None:
                chunk_labels = fetch_chunk_frames(lr_cap,
                                                  cnfg['n_chunk_frames'],
                                                  cnfg['step_frame'],
                                                  None if cnfg['lr_vid_format'] == 'yuv' else cnfg['lr_size'])
            else:
                chunk_labels = [cv2.resize(hrf, cnfg['lr_size'], interpolation=cv2.INTER_AREA) for hrf in chunk_frames]

            chunk_frames = np.array(chunk_frames).astype(np.float32)
            chunk_labels = np.array(chunk_labels).astype(np.float32)
            chunk_frames_norm = (chunk_frames / 127.5) - 1.0
            chunk_labels_norm = (chunk_labels / 127.5) - 1.0

        if args.inference:
            r = total_frames % chunk_frames.shape[0]
        else:
            r = np.random.randint(chunk_frames.shape[0])

        hr_norm = chunk_frames_norm[r]
        lr_norm = chunk_labels_norm[r]
        total_frames += 1
        yield (lr_norm, hr_norm, chunk_no)
    hr_cap.release()
    if lr_cap is not None:
        lr_cap.release()


def augment(image, label, chunk_no):
    if args.crop_augment:
        h = cnfg['lr_size'][1]
        w = cnfg['lr_size'][0]
        in_crop_height = h // 2
        in_crop_width = w // 2
        scale = 4
        in_x = tf.random.uniform(shape=(), maxval=w - in_crop_width, dtype=tf.int32)
        in_y = tf.random.uniform(shape=(), maxval=h - in_crop_height, dtype=tf.int32)
        cropped_image = image[:, in_y:in_y + in_crop_height, in_x:in_x + in_crop_width]
        cropped_label = label[:, scale * in_y:scale * (in_y + in_crop_height),
                        scale * in_x:scale * (in_x + in_crop_width)]
        return cropped_image, cropped_label, chunk_no
    else:
        return image, label, chunk_no


# Read the video information
cnfg = get_config()
fps = cnfg['fps']

# DNN model
# Create the data pipeline
dataset = tf.data.Dataset.from_generator(gen_video,
                                         output_types=(tf.float32, tf.float32, tf.int32),
                                         output_shapes=(tf.TensorShape([None, None, 3]),
                                                        tf.TensorShape([None, None, None]),
                                                        tf.TensorShape([])))
dataset = dataset.batch(args.batch_size).map(augment).repeat(1).prefetch(100)
iter = dataset.make_one_shot_iterator()
lr, hr, chunk_no = iter.get_next()

# Super-resolve
sr = getattr(models, args.model_name)(lr)

# Bicubic baseline
cubic_restore = tf.image.resize_bicubic(lr,
                                        (tf.shape(hr)[1], tf.shape(hr)[2]),
                                        align_corners=False,
                                        half_pixel_centers=True)

# Define the loss
loss = tf.losses.absolute_difference(sr, hr) if args.l1_loss else tf.losses.mean_squared_error(sr, hr)

# Fractional updates modules
tvars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(args.lr)
update_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if args.coord_frac < 1.0:
    var_names = [v.name for v in tvars]
    grad_masks_pl = {v.name: tf.placeholder(shape=v.shape, dtype=tf.bool, name='%s_maskpl' % v.name.rstrip(':0')) for v
                     in tvars}
    backup_vars = {v.name: tf.Variable(tf.zeros_like(v), name='%s_copy' % v.name.rstrip(':0'), trainable=False) for v in
                   tvars}
    main_vars = {v.name: v for v in tvars}
    with tf.control_dependencies(update_bn):
        backup_ops = [tf.assign(backup_vars[k], main_vars[k], use_locking=True) for k in backup_vars]
    with tf.control_dependencies(backup_ops):
        train_all = optimizer.minimize(loss)
    with tf.control_dependencies([train_all]):
        train = [tf.assign(main_vars[k], tf.where(grad_masks_pl[k], main_vars[k], backup_vars[k]), use_locking=True) for
                 k in main_vars]
else:
    with tf.control_dependencies(update_bn):
        train = optimizer.minimize(loss)

# Summary metrics
sr_cut = tf.clip_by_value(sr, -1, 1)
tf.summary.image('lr', lr)
tf.summary.image('sr', sr_cut)
tf.summary.image('bicubic', cubic_restore)

rgb_sr_01 = tf.clip_by_value(0.5 * sr_cut + 0.5, 0, 1)
rgb_hr_01 = tf.clip_by_value(0.5 * hr + 0.5, 0, 1)
rgb_cubic_01 = tf.clip_by_value(0.5 * cubic_restore + 0.5, 0, 1)
model_metrics = get_metrics(rgb_sr_01, rgb_hr_01)
cubic_metrics = get_metrics(rgb_cubic_01, rgb_hr_01)

init = tf.initializers.global_variables()
saver = tf.train.Saver(var_list=tvars)
manual_saver = save_helper(graph=tf.get_default_graph(), map_fun=lambda x: x)
sum_writer = tf.summary.FileWriter('%s/summary/log' % args.save_path)
tf.summary.scalar('total_loss', loss)
tf.summary.scalar('model_psnr_y', model_metrics['psnr_y'])
tf.summary.scalar('cubic_psnr_y', cubic_metrics['psnr_y'])
tf.summary.scalar('psnr_y_diff', model_metrics['psnr_y'] - cubic_metrics['psnr_y'])
tf.summary.scalar('model_ssim', model_metrics['ssim'])
summary = tf.summary.merge_all()


def gradient_guided_params(before, after, coord_frac):
    changes = [np.reshape(np.abs(after[v] - before[v]), (-1,)) for v in var_names]
    changes = np.concatenate(changes, axis=0)
    cut_threshold = np.percentile(changes, 100 * (1 - coord_frac))
    train_mask = {grad_masks_pl[v]: np.abs(after[v] - before[v]) > cut_threshold for v in var_names}
    n_selected_coords = np.sum([np.sum(train_mask[grad_masks_pl[v]]) for v in var_names])
    print('Selected %d parameters for coordinate descent' % n_selected_coords)
    return train_mask


def inference_loop(sess):
    image_path = '%s/images' % args.save_path
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    model_record = []
    cubic_record = []

    if args.single_checkpoint:
        chunk_restore_from = args.load_path
        saver.restore(sess, chunk_restore_from)
        print('Restored %s' % chunk_restore_from)

    for i in range(cnfg['num_frames']):
        if i % cnfg['n_chunk_frames'] == 0:
            chunk_no_ = (i * args.batch_size) // cnfg['n_chunk_frames']
            restore_chunk_id = chunk_no_ - 1 if args.online else chunk_no_
            if not args.single_checkpoint:
                chunk_restore_from = '%s/checkpoints/chunk_%06d/model' % (args.load_path, restore_chunk_id)
                saver.restore(sess, chunk_restore_from)
                print('Restored %s' % chunk_restore_from)

        if not i % cnfg['n_chunk_frames'] == 0:  # save one sample per chunk
            actual_chunk_no_, model_metrics_, cubic_metrics_ = sess.run([chunk_no, model_metrics, cubic_metrics])
        elif args.dump_samples:
            rgb_sr_01_, rgb_hr_01_, rgb_cubic_01_, actual_chunk_no_, model_metrics_, cubic_metrics_ = sess.run(
                    [rgb_sr_01, rgb_hr_01, rgb_cubic_01, chunk_no, model_metrics, cubic_metrics])
            np.save('%s/rec_%06d.npy' % (image_path, i), np.array(rgb_sr_01_))
            np.save('%s/org_%06d.npy' % (image_path, i), np.array(rgb_hr_01_))
            np.save('%s/cub_%06d.npy' % (image_path, i), np.array(rgb_cubic_01_))
        assert actual_chunk_no_ == chunk_no_

        cubic_record.append(cubic_metrics_)
        model_record.append(model_metrics_)
        if i % 100 == 0:
            print('frame=%d' % i)
            print('model_metrics=', model_metrics_)
            print('cubic_metrics=', cubic_metrics_)
            with open(image_path + '/cubic_result.pkl', 'wb') as f:
                pickle.dump(cubic_record, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('pickle dumped')
            with open(image_path + '/model_result.pkl', 'wb') as f:
                pickle.dump(model_record, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('pickle dumped')


def training_loop(sess):
    chunk_no_ = args.ff_nchunks - 1
    i = args.ff_nchunks * cnfg['n_chunk_iterations'] - 1
    print('----> Fast-forwarding %d chunks' % args.ff_nchunks)
    if not args.load_path == '':
        saver.restore(sess, args.load_path)
        print('----> Restored from %s' % args.load_path)

    feed_dict = {}
    dt = 0
    train_iter_time = deque(maxlen=100)
    while True:
        i += 1
        if i % cnfg['n_chunk_iterations'] == 0:
            chunk_save_to = '%s/checkpoints/chunk_%06d/' % (args.save_path, chunk_no_)
            os.makedirs(chunk_save_to, exist_ok=True)
            saver.save(sess, chunk_save_to + '/model')
            print('Saved the model to %s' % chunk_save_to)
            chunk_no_ += 1
            if chunk_no_ == cnfg['total_chunks']:
                break
            if args.coord_frac < 1.0:
                # Backup the model
                before = manual_saver.save_vars(sess, tvars, lambda x: x)
                # Apply full gradient descent steps
                for warmup_step in range(cnfg['n_chunk_frames'] * cnfg['warmup_epochs']):
                    sess.run(train_all)

                after = manual_saver.save_vars(sess, tvars, lambda x: x)
                train_mask = gradient_guided_params(before, after, args.coord_frac)
                # Restore the entire model to the most recent chunk
                saver.restore(sess, chunk_save_to + 'model')
                for k in train_mask:
                    feed_dict[k] = train_mask[k]

        if i % 500 == 0:
            t0 = time.time()
            res_ = sess.run(
                    {'loss': loss, 'train': train, 'model_metrics': model_metrics, 'cubic_metrics': cubic_metrics},
                    feed_dict)
            t1 = time.time()
            print('bechmarking_time=%.3f, mean_train_iter_time=%.3f sec' % (t1 - t0, np.mean(train_iter_time)))
            print('Chunk: %d, Iteration: %d, Loss: %f, inference time: %.3f' % (
                    chunk_no_, i, res_['loss'], dt))
            print('Bicubic: ', res_['cubic_metrics'])
            print('Model:', res_['model_metrics'])
        else:
            t0 = time.time()
            _, actual_chunk_no_ = sess.run([train, chunk_no], feed_dict)
            t1 = time.time()
            train_iter_time.append(t1 - t0)
            assert (actual_chunk_no_ == chunk_no_).all()


# Train and infer
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.14)
config = tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    sess.run(init)
    if args.inference:
        inference_loop(sess)
    else:
        training_loop(sess)
