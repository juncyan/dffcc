import datetime
import os
import argparse
import torch
from .logger import load_logger

__all__ = ['Config', 'parse_args']

def Config(args, model_name='RSICC'):
    
    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    save_dir = os.path.join(args.dst_dir, f"{args.data_name}/{model_name}_{time_flag}")
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_path = os.path.join(save_dir, "train_{}_{}.log".format(model_name, args.data_name[:10]))
    
    args.best_model_path = os.path.join(save_dir, "{}_{}_best.pth.tar".format(model_name, args.data_name[:10]))
    args.metric_path = os.path.join(save_dir, "{}_{}_metrics.csv".format(model_name, args.data_name[:10]))
    args.save_dir = save_dir
    args.caption_res = os.path.join(save_dir, f"{model_name}_res.json")
    args.caption_gts = os.path.join(save_dir, f"{model_name}_gts.json")
    args.logger = load_logger(log_path)
    args.logger.info(f"{model_name}")
    return args

def save_checkpoint(args, epoch, epochs_since_improvement, encoder_image,encoder_feat, decoder,
                    encoder_image_optimizer,encoder_feat_optimizer, decoder_optimizer, bleu4):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder_image': encoder_image,
             'encoder_feat': encoder_feat,
             'decoder': decoder,
             'encoder_image_optimizer': encoder_image_optimizer,
             'encoder_feat_optimizer': encoder_feat_optimizer,
             'decoder_optimizer': decoder_optimizer,
             }
    
    torch.save(state, args.best_model_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='msfgnet',
                        help='model name (default: msfgnet)')
    parser.add_argument('--device', type=str, default='gpu:0',
                        choices=['gpu:0', 'gpu:1', 'cpu'],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default='LEVIR_CD',
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--img_ab_concat', type=bool, default=False,
                        help='img_ab_concat False')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num classes (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size (default: 4)')
    
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    
    parser.add_argument('--num_works', type=int, default=8,
                        help='num_works (default: 8)')
    args = parser.parse_args()
    return args