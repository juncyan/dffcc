import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from core.dataloader import CaptionDataset
from .utils import *
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

def save_captions(args, word_map, hypotheses, references):
    result_json_file = {}
    reference_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            # args.logger.info(word)
            line_hypo += word[0] + " "

        result_json_file[str(kkk)] = []
        result_json_file[str(kkk)].append(line_hypo)

        line_hypo += "\r\n"

    kkk = -1
    for item in tqdm(references):
        kkk += 1

        reference_json_file[str(kkk)] = []

        for sentence in item:
            line_repo = ""
            for word_idx in sentence:
                word = get_key(word_map, word_idx)
                line_repo += word[0] + " "
            reference_json_file[str(kkk)].append(line_repo)
            line_repo += "\r\n"

    with open(args.caption_res, 'w') as f:
        json.dump(result_json_file, f)

    with open(args.caption_gts, 'w') as f:
        json.dump(reference_json_file, f)

def save_captions_blue4(args, word_map, hypotheses, references, blue4):
    result_json_file = {}
    reference_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            # args.logger.info(word)
            line_hypo += word[0] + " "

        result_json_file[str(kkk)] = []
        result_json_file[str(kkk)].append(line_hypo)

        line_hypo += "\r\n"

    kkk = -1
    for item in tqdm(references):
        kkk += 1

        reference_json_file[str(kkk)] = []

        for sentence in item:
            line_repo = ""
            for word_idx in sentence:
                word = get_key(word_map, word_idx)
                line_repo += word[0] + " "
            reference_json_file[str(kkk)].append(line_repo)
            line_repo += "\r\n"

    res_path = args.caption_res[:-5] + f"_blue4{blue4:.4f}.json"
    gt_path = args.caption_gts[:-5] + f"_blue4{blue4:.4f}.json"
    with open(res_path, 'w') as f:
        json.dump(result_json_file, f)

    with open(gt_path, 'w') as f:
        json.dump(reference_json_file, f)
        
def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def write_metrics(metrics, cls_metrics, args):
    args.logger.info(cls_metrics)
    args.logger.info("BLEU-1 {:.4f} BLEU-2 {:.4f} BLEU-3 {:.4f} BLEU-4 {:.4f} ".format
              (metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"], metrics["Bleu_4"]))
    args.logger.info("METEOR {:.4f} ROUGE_L {:.4f} CIDEr {:.4f}".format
              (metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

def evaluate_transformer(args, encoder_image, encoder_feat, decoder):
    # Load model
    encoder_image = encoder_image.to(args.device)
    encoder_image.eval()
    encoder_feat = encoder_feat.to(args.device)
    encoder_feat.eval()
    decoder = decoder.to(args.device)
    decoder.eval()

    # Load word map (word2ix)
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    """
    Evaluation for decoder: transformer
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, args.Split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc=0
    nochange_acc=0

    with torch.no_grad():
        for i, (image_pairs, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc=args.Split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            # 5 image is the same when "shuffle=False" of the dataloader
            if (i + 1) % 5 != 0:
                continue
            # if i>10:
            #     break
            k = beam_size

            # Move to GPU device, if available
            image_pairs = image_pairs.to(args.device)  # [1, 2, 3, 256, 256]

            # Encode
            imgs_A = image_pairs[:, 0, :, :, :]
            imgs_B = image_pairs[:, 1, :, :, :]
            # imgs_A = encoder_image(imgs_A)
            # imgs_B = encoder_image(imgs_B)  # encoder_image :[1, 1024,14,14]
            # encoder_out = encoder_feat(imgs_A, imgs_B) # encoder_out: (S, batch, feature_dim)
            
            img_cat = torch.cat((imgs_A, imgs_B), 1)
            feature, yl = encoder_image(img_cat)   
            encoder_out = encoder_feat(yl, feature)
            
            tgt = torch.zeros(52, k).to(args.device).to(torch.int64)
            tgt_length = tgt.size(0)
            mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.to(args.device)

            tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).to(args.device) # k_prev_words:[52,k]
            # Tensor to store top k sequences; now they're just <start>
            seqs = torch.LongTensor([[word_map['<start>']]*1] * k).to(args.device)  # [1,k]
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(args.device)
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            k_prev_words = tgt.permute(1,0)
            S = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            # # We'll treat the problem as having a batch size of k, where k is beam_size
            encoder_out = encoder_out.expand(S,k, encoder_dim)  # [S,k, encoder_dim]
            encoder_out = encoder_out.permute(1,0,2)

            # Start decoding
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                tgt = k_prev_words.permute(1,0)
                tgt_embedding = decoder.vocab_embedding(tgt)
                tgt_embedding = decoder.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)  # (length, batch, feature_dim)
                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.wdc(pred)  # (length, batch, vocab_size)
                scores = pred.permute(1,0,2)  # (batch,length,  vocab_size)
                scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)


                # Convert unrolled indices to actual indices of scores
                # prev_word_inds = top_k_words // vocab_size  # (s)
                # if max(top_k_words)>vocab_size:
                #     args.logger.info(">>>>>>>>>>>>>>>>>>")
                prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # Important: this will not work, since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step + 1] = seqs  # [s, 52]
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1


            # choose the caption which has the best_score.
            if (len(complete_seqs_scores) == 0):
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            if (len(complete_seqs_scores) > 0):
                assert Caption_End
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[indices]
                # References
                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads

                references.append(img_captions)
                # Hypotheses
                new_sent = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                hypotheses.append(new_sent)
                assert len(references) == len(hypotheses)

                # # 判断有没有变化
                nochange_list = ["the scene is the same as before ", "there is no difference ",
                                 "the two scenes seem identical ", "no change has occurred ",
                                 "almost nothing has changed "]
                ref_sentence = img_captions[1]
                ref_line_repo = ""
                for ref_word_idx in ref_sentence:
                    ref_word = get_key(word_map, ref_word_idx)
                    ref_line_repo += ref_word[0] + " "

                hyp_sentence = new_sent
                hyp_line_repo = ""
                for hyp_word_idx in hyp_sentence:
                    hyp_word = get_key(word_map, hyp_word_idx)
                    hyp_line_repo += hyp_word[0] + " "
                # 对于变化图像对
                if ref_line_repo not in nochange_list:
                    change_references.append(img_captions)
                    change_hypotheses.append(new_sent)
                    if hyp_line_repo not in nochange_list:
                        change_acc = change_acc+1
                else:
                    nochange_references.append(img_captions)
                    nochange_hypotheses.append(new_sent)
                    if hyp_line_repo in nochange_list:
                        nochange_acc = nochange_acc+1

        # captions
        # save_captions(args, word_map, hypotheses, references)

    args.logger.info('len nochange_references: {}'.format(len(nochange_references)))
    args.logger.info('len change_references: {}'.format(len(change_references)))
    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    nochange_metric = None
    change_metric = None
    if len(nochange_references)>0:
        args.logger.info('nochange_metric:')
        nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
        args.logger.info("nochange_acc:{:.4f}".format(nochange_acc / len(nochange_references)))
    if len(change_references)>0:
        args.logger.info('change_metric:')
        change_metric = get_eval_score(change_references, change_hypotheses)
        args.logger.info("change_acc: {:.4f}".format(change_acc / len(change_references)))
    args.logger.info(".......................................................")

    metrics = get_eval_score(references, hypotheses)
    if nochange_metric is not None:
        write_metrics(nochange_metric, "nochange metrics", args)
        args.logger.info(".......................................................")
    if change_metric is not None:
        write_metrics(change_metric, "change metrics", args)
        args.logger.info(".......................................................")
    write_metrics(metrics, "all metrics", args)
    args.logger.info(".......................................................")
    save_captions_blue4(args, word_map, hypotheses, references, metrics["Bleu_4"])
    
    return metrics, nochange_metric, change_metric

