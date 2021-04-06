import numpy as np
import tensorflow as tf


def get_metrics(im1, im2):
    # Assumes im1 and im2 are in RGB space with float values in range [0, 1]
    im1 = tf.clip_by_value(im1, 0, 1)
    im2 = tf.clip_by_value(im2, 0, 1)
    im1_y = tf.image.rgb_to_yuv(im1)[...,0]
    im2_y = tf.image.rgb_to_yuv(im2)[...,0]
    psnr_y = tf.reduce_mean(tf.image.psnr(im1_y, im2_y, max_val=1.))
    psnr_rgb = tf.reduce_mean(tf.image.psnr(im1, im2, max_val=1.))
    ssim = tf.reduce_mean(tf.image.ssim(im1, im2, max_val=1.))
    metrics = {'ssim': ssim, 'psnr_y': psnr_y, 'psnr_rgb': psnr_rgb}
    return metrics

class SaveHelper:
    def __init__(self, graph, map_fun):
        with graph.as_default():
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.vars_dict = {map_fun(v.name): v for v in variables}
            self.vars_pl = {
                map_fun(v.name): tf.placeholder(dtype=v.dtype, shape=v.shape, name='pl_%s' % (v.name.strip(':0'))) for v
                in variables}
            self.load_ops = {k: tf.assign(self.vars_dict[k], self.vars_pl[k], use_locking=True) for k in self.vars_dict}

    def save_vars(self, sess, vars_list, map_fun, save_dir=None):
        vars_vals = sess.run(vars_list)
        save_dict = {}
        for i in range(len(vars_list)):
            if map_fun(vars_list[i].name):
                save_dict[map_fun(vars_list[i].name)] = vars_vals[i]
        if save_dir:
            np.save(save_dir, save_dict)
        return save_dict

    def restore_vars(self, sess, load_dir, map_fun):
        print('Trying to restore checkpoint')
        if isinstance(load_dir, str):
            vars_list = np.load(load_dir, allow_pickle=True).item()
        elif isinstance(load_dir, dict):
            vars_list = load_dir
        else:
            exit(1)
        restore_ops = [self.load_ops[var_name] for var_name in vars_list if not map_fun(var_name) is None]
        feed_dict = {self.vars_pl[var_name]: vars_list[var_name] for var_name in vars_list if
                     not map_fun(var_name) is None}
        sess.run(restore_ops, feed_dict=feed_dict)
        print('Restored successfully')
        return
