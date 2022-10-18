# -*- coding: utf-8 -*-
# Copyright 2021 Andre Guarda
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Adaptive Deep Learning-based Point Cloud Coding (ADL-PCC).

This is the software of the codec published in:
A. Guarda, Nuno M. M. Rodrigues and F. Pereira,
“Adaptive Deep Learning-based Point Cloud Geometry Coding,”
in IEEE Journal on Selected Topics in Signal Processing (J-STSP),
vol. 15, no. 2, pp. 415–430, Italy, Feb. 2021.
doi: 10.1109/JSTSP.2020.3047520.

 
The DL coding model is based on the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os
import numpy as np
import pickle
import gzip
import shutil

from absl import app
from absl.flags import argparse_flags

import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

import loss_functions
import pc2vox
from transforms import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform, HyperSynthesisTransform

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

# tf.config.optimizer.set_jit(True)
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# function to load data sequentially
def load_data(path):
    print(path)
    filename = tf.strings.as_string(path)
    print(filename)
    data = np.load(filename) # read compressed npy
    return data_tensor

def _set_shapes(data):
    data.set_shape((64,64,64,1))
    return data

def placeholder_func(vox_data, b_size):
    np.random.shuffle(vox_data)
    train_data_placeholder = tf.placeholder(vox_data.dtype, vox_data.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data_placeholder)
    train_dataset = train_dataset.shuffle(buffer_size=len(vox_data)).repeat()
    train_dataset = train_dataset.batch(b_size)
    train_dataset = train_dataset.prefetch(16)

    return train_dataset, train_data_placeholder

def read_npy_file(item):
    data = np.load(item.decode())
    data.set_shape((64,64,64,1))
    print("************************************",data)
    return data.astype(np.float32) 

def tf_data_generator(d_path, file_list, i, batch_size = 4):
    if i%(np.floor(len(file_list)/batch_size)-1) == 0:
        if i == 0:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size]
        else:
            count = int(np.floor(len(file_list)/batch_size)-1)
            file_chunk = file_list[count*batch_size:len(file_list)]
            print("**************Epoch**************")
            np.random.shuffle(file_list)

    else:
        count = int(i%(np.floor(len(file_list)/batch_size)))
        file_chunk = file_list[count*batch_size:(count+1)*batch_size] 
        
    data = []
    for file in file_chunk:
        temp = np.load(os.path.join(d_path,file))
        data.append(temp)

    data = np.asarray(data).reshape(-1,64,64,64,1)
    return data


def train(args):

    """Trains the model."""   
    path = args.train_data
    data_files = os.listdir(path)
    np.random.shuffle(data_files)
    print(len(data_files))
    ite = 0

    # each numpy array is of shape (64, 64, 64, 1)
    train_data_placeholder = tf.placeholder('float32', [args.batchsize, 64, 64, 64, 1])

    # Instantiate model
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Build autoencoder and hyperprior
    y = analysis_transform(train_data_placeholder)
    z = hyper_analysis_transform(abs(y))
    z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
    sigma = hyper_synthesis_transform(z_tilde)
    scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
    y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde)

    # Compute distortion: Focal Loss
    train_mse = loss_functions.focal_loss(train_data_placeholder, x_tilde, gamma=args.fl_gamma, alpha=args.fl_alpha)

    # Compute rate: Total number of bits divided by number of points
    num_input_points = tf.reduce_sum(train_data_placeholder)
    train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) + tf.reduce_sum(tf.log(z_likelihoods))) / (
                -np.log(2) * num_input_points)

    # Compute the rate-distortion cost
    train_loss = train_mse + (args.lmbda * train_bpp)

    # Minimize loss and auxiliary loss, and execute update op
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    # Write summaries for Tensorboard visualization
    rec_count_real = tf.reduce_sum(tf.cast(tf.greater_equal(x_tilde, 0.5), tf.float32))
    count_ratio = rec_count_real / num_input_points
    tf.summary.scalar("1_loss", train_loss)
    tf.summary.scalar("2_mse", train_mse)
    tf.summary.scalar("3_bpp", train_bpp)
    tf.summary.scalar("4_count_ratio", tf.reduce_mean(count_ratio))
    tf.summary.scalar("5_count_in_real", tf.reduce_mean(num_input_points))
    tf.summary.scalar("6_count_out_real", tf.reduce_mean(rec_count_real))
    tf.summary.histogram("x_tilde", x_tilde)
    tf.summary.histogram("y_tilde", y_tilde)
    tf.summary.histogram("y", y)

    g_step = tf.train.get_or_create_global_step()
    writer = tf.summary.FileWriter(args.checkpoint_dir)
    summaries = tf.summary.merge_all()

    #sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    init_op = tf.global_variables_initializer()

    if os.path.isfile(os.path.join(args.checkpoint_dir, 'model.ckpt.meta')):
        with tf.Session() as sess:
            #sess.run([writer.init(), step.initializer])
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1, save_relative_paths=True)
            new_saver = tf.train.import_meta_graph(os.path.join(args.checkpoint_dir, 'model.ckpt.meta'))
            new_saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
            #sess.run(x_data_iterator.initializer)
            sess.run(init_op)
            stat_g_step = sess.run(g_step)
            for i in range(stat_g_step, args.last_step):
                x_data = tf_data_generator(path, data_files, i, args.batchsize)
                feed_dict = {train_data_placeholder: x_data}
                sess.run(train_op, feed_dict=feed_dict)
                if i%1000==0:
                    print("Iter: ", i)
                    summ = sess.run(summaries, feed_dict=feed_dict)
                    writer.add_summary(summ, global_step=i)
                    saver.save(sess, os.path.join(args.checkpoint_dir,'model.ckpt'), write_meta_graph=True)    
                
    else:
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1, save_relative_paths=True)
        with tf.Session() as sess:
            sess.run(init_op)
            #sess.run(x_data_iterator.initializer) 
            for i in range(args.last_step):
                x_data = tf_data_generator(path, data_files, i, args.batchsize)
                feed_dict = {train_data_placeholder: x_data}
                sess.run(train_op, feed_dict=feed_dict)
                if i%1000==0:
                    print("Iter: ", i)
                    summ = sess.run(summaries, feed_dict=feed_dict )
                    writer.add_summary(summ, global_step=i)
                    saver.save(sess, os.path.join(args.checkpoint_dir,'model.ckpt'), write_meta_graph=True)
                    
                
                           


def compress(args):
    """Compresses all test voxel_blocks."""

    x = tf.placeholder(tf.float32, [None, None, None, None, 1])

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Transform and compress the voxel block
    y = analysis_transform(x)
    y_shape = tf.shape(y)

    z = hyper_analysis_transform(abs(y))
    z_hat, z_likelihoods = entropy_bottleneck(z, training=False)

    sigma = hyper_synthesis_transform(z_hat)
    sigma = sigma[:, :y_shape[1], :y_shape[2], :y_shape[3], :]
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
    side_string = entropy_bottleneck.compress(z)
    string = conditional_bottleneck.compress(y)

    y_hat = conditional_bottleneck.decompress(string)
    x_hat = synthesis_transform(y_hat)

    tensors = [string, side_string]

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))

    with tf.Session(config=sess_config) as sess:
        # Manage input and output directories
        in_file = args.input_file
        if not in_file.endswith('.ply'):
            raise ValueError("Input must be a PLY file (.ply extension).")

        pc_filename = os.path.splitext(os.path.basename(in_file))[0]
        stream_dir = os.path.join("/media/extdrive/rohith/", "results_dynamic_data_batch16", os.path.split(os.path.split(args.checkpoint_dir)[0])[1], pc_filename)
        os.makedirs(stream_dir, exist_ok=True)

        # Load input PC, get list of coordinates
        in_points = pc2vox.load_pc(in_file)
        print(len(in_points))
        # Divide PC into blocks of the desired size. Get list of relative coordinates for points in each block
        blocks, blk_map = pc2vox.pc2blocks(in_points, args.blk_size)

        # Get the different models directory names
        model_names = glob.glob(args.checkpoint_dir)

        total_cost = np.zeros([len(blocks), len(model_names)], np.float)
        total_bitstream = []

        # Iterate each model
        for j in range(len(model_names)):
            bitstream = []
            # Load the latest model checkpoint
            latest = tf.train.latest_checkpoint(checkpoint_dir=model_names[j])
            tf.train.Saver().restore(sess, save_path=latest)

            try:
                # Iterate all blocks
                for i in range(len(blocks)):
                    temp_blk = blocks[i]
                    num_blk_points = temp_blk.shape[0]
                    # Encode and decode block
                    arrays, x_rec = sess.run([tensors, x_hat], feed_dict={x: pc2vox.point2vox(temp_blk, args.blk_size)})
                    # Compute block bitrate
                    packed = tfc.PackedTensors()
                    packed.pack(tensors, arrays)
                    bpp = len(packed.string) * 8 / num_blk_points
                    # Compute block RD cost
                    mse = loss_functions.point2point(temp_blk, pc2vox.vox2point(np.greater_equal(np.squeeze(x_rec), 0.5)))
                    total_cost[i, j] = mse + (args.lmbda * bpp)

                    bitstream.extend([packed.string])

                total_bitstream.extend([bitstream])

            except tf.errors.OutOfRangeError:
                pass

        best_model = np.argmin(total_cost, axis=1)

        final_bitstream = [total_bitstream[best_model[i]][i] for i in range(len(blocks))]

        with open(os.path.join(stream_dir, pc_filename + ".pkl"), "wb") as f:
            pickle.dump([args.blk_size, best_model, blk_map, final_bitstream], f)

        with open(os.path.join(stream_dir, pc_filename + ".pkl"), 'rb') as f_in:
            with gzip.open(os.path.join(stream_dir, pc_filename + ".pkl.gz"), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(os.path.join(stream_dir, pc_filename + ".pkl"))



def decompress(args):
    """Decompresses all test voxel_blocks."""
    # Read the shape information and compressed string from the binary file
    string = tf.placeholder(tf.string, [1])
    side_string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [3])
    y_shape = x_shape // 8
    z_shape = y_shape // 4

    # Instantiate model
    synthesis_transform = SynthesisTransform(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

    # Decompress and transform the voxel block back
    z_hat = entropy_bottleneck.decompress(
        side_string, tf.concat([z_shape, [args.num_filters]], axis=0), channels=args.num_filters)

    sigma = hyper_synthesis_transform(z_hat)
    sigma = sigma[:, :y_shape[0], :y_shape[1], :y_shape[2], :]
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, dtype=tf.float32)
    y_hat = conditional_bottleneck.decompress(string)
    x_hat = synthesis_transform(y_hat)

    tensors = [string, side_string]

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))

    print("*****************opening session***************")
    with tf.Session(config=sess_config) as sess:
        # Get the different models directory names
        model_names = glob.glob(args.checkpoint_dir)
        print(model_names)

        # Manage input and output directories
        stream_filename = args.input_file
        if not stream_filename.endswith('.gz'):
            raise ValueError("Input bitstream file must have .gz extension.")

        with open(stream_filename + ".dec.pkl", 'wb') as f_out:
            with gzip.open(stream_filename, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)

        with open(stream_filename + ".dec.pkl", "rb") as f:
            blk_size, best_model, blk_map, final_bitstream = pickle.load(f)
        print(best_model)
        
        try:
            # Initialize the reconstructed PC (empty)
            pts_geom = np.array([], dtype=np.int32).reshape(0, 3)

            # Iterate each model
            for j in range(len(model_names)):
                # Load the latest model checkpoint
                latest = tf.train.latest_checkpoint(checkpoint_dir=model_names[j])
                tf.train.Saver().restore(sess, save_path=latest)
                
                # Iterate all blocks
                for i in range(len(best_model)):
                    if best_model[i] == j:
                        print("1")
                        # Unpack string corresponding to the coded block
                        packed = tfc.PackedTensors(final_bitstream[i])
                        arrays = packed.unpack(tensors)
                        
                        # Decode block
                        x_rec = sess.run(x_hat, feed_dict=dict(zip(tensors + [x_shape], arrays + [[blk_size, blk_size, blk_size]])))
                        #print(x_rec)
                        # Convert back to point coordinates
                        points = pc2vox.vox2point(np.greater_equal(np.squeeze(x_rec), 0.5))
                        print(points)
                        points = points + (blk_size * blk_map[i])
                        print(points)
                        # Merge block points in the fully reconstructed PC
                        pts_geom = np.concatenate((pts_geom, points))
                        print("5")

            print(pts_geom)            
            # Write reconstructed PC to file
            pc2vox.save_pc(pts_geom, stream_filename + ".dec.ply")
            os.remove(stream_filename + ".dec.pkl")
            print("7")
        except tf.errors.OutOfRangeError:
            pass


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--num_filters", type=int, default=32,
        help="Number of filters in each convolutional layer.")
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains a new model."
             "'compress' reads the test PC file and writes a compressed binary stream."
             "'decompress' reads the binary stream and reconstructs the PC."
             "input filenames need to be provided. Invoke '<command> -h' for more information.")

    # 'train' subcommand
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains a new model.")
    train_cmd.add_argument(
        "train_data",
        help="Directory containing PC data for training. Filenames should be provided with"
             " a glob pattern that expands into a list of PCs: data/*.ply ")
    train_cmd.add_argument(
        "checkpoint_dir", 
        help="Directory where to save model checkpoints. "
             "For training, a single directory should be provided: ../models/test ")
    train_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    train_cmd.add_argument(
        "--last_step", type=int, default=1000000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--lambda", type=float, default=1000, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--fl_alpha", type=float, default=0.75,
        help="Class balancing weight for Focal Loss.")
    train_cmd.add_argument(
        "--fl_gamma", type=float, default=2.0,
        help="Focusing weight for Focal Loss.")

    # 'compress' subcommand
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a Point Cloud PLY file, compresses it, and writes the bitstream file.")
    compress_cmd.add_argument(
        "input_file",
        help="Input Point Cloud filename (.ply).")
    compress_cmd.add_argument(
        "checkpoint_dir",
        help="Directory where to load model checkpoints."
             "For compression, a glob pattern that expands into a list of directories"
             "(each corresponding to a different trained DL coding model) should be provided: ../models/* ")
    compress_cmd.add_argument(
        "--blk_size", type=int, default=64,
        help="Size of the 3D coding block units.")
    compress_cmd.add_argument(
        "--lambda", type=float, default=0, dest="lmbda",
        help="Lambda for RD trade-off when selecting best DL coding model.")

    # 'decompress' subcommand
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a bitstream file, decodes the voxel blocks, and reconstructs the Point Cloud.")
    decompress_cmd.add_argument(
        "input_file",
        help="Input bitstream filename (.gz).")
    decompress_cmd.add_argument(
        "checkpoint_dir",
        help="Directory where to load model checkpoints."
             "For decompression, a glob pattern that expands into a list of directories"
             "(each corresponding to a different trained DL coding model) should be provided: ../models/* ")

    # Parse arguments
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand
    if not args.checkpoint_dir:
        raise ValueError("Need checkpoint directory to save or load model.")
    if args.command == "train":
        if not args.train_data:
            raise ValueError("Need input PC filenames for training.")
        train(args)
    elif args.command == "compress":
        if not args.input_file:
            raise ValueError("Need input PC filename for encoding.")
        compress(args)
    elif args.command == "decompress":
        if not args.input_file:
            raise ValueError("Need input bitstream filename for decoding.")
        decompress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
