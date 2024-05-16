import glob
import pickle

import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
import argparse
from PIL import Image


def dice(gt,pred):
    if (pred.sum() + gt.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum())


def hd(gt,pred):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0
    

def iou(gt,pred):
  """
  计算预测与ground truth之间的IoU
  """

  intersection = np.logical_and(pred, gt)
  union = np.logical_or(pred, gt)
  iou = np.sum(intersection) / np.sum(union)
  
  return iou

def test_two(args):
    mask_list = sorted(glob.glob(os.path.join(args.save_dir, 'mask', '*png')))
    pre_mask_list = sorted(glob.glob(os.path.join(args.save_dir, 'pre_mask', '*png')))

    Dice_in = []
    Dice_out = []

    hd_in = []
    hd_out = []

    iou_in = []
    iou_out = []

    def process_label(label):
        # mask_in = label == 160
        mask_out = label == 80
        return mask_out
    
    fw = open(args.save_dir + '/dice_pre.txt', 'a')
    for mask_path, pre_mask_path in zip(mask_list, pre_mask_list):
        # print(mask_path.split('/')[-1])
        # print(pre_mask_path.split('/')[-1])
        mask = np.array(Image.open(mask_path))
        pre_mask = np.array(Image.open(pre_mask_path))

        mask_in = process_label(mask)
        pre_mask_in = process_label(pre_mask)

        Dice_in.append(dice(mask_in, pre_mask_in))
        # Dice_out.append(dice(mask_out, pre_mask_out))

        hd_in.append(hd(mask_in, pre_mask_in))
        # hd_out.append(hd(mask_out, pre_mask_out))

        iou_in.append(iou(mask_in, pre_mask_in))
        # iou_out.append(iou(mask_out, pre_mask_out))

        fw.write('*' * 20 + '\n', )
        fw.write(pre_mask_path.split('/')[-1] + '\n')
        fw.write('Dice_in: {:.4f}\n'.format(Dice_in[-1]))
        # fw.write('Dice_out: {:.4f}\n'.format(Dice_out[-1]))
        fw.write('hd_in: {:.4f}\n'.format(hd_in[-1]))
        # fw.write('hd_out: {:.4f}\n'.format(hd_out[-1]))
        fw.write('iou_in: {:.4f}\n'.format(iou_in[-1]))
        # fw.write('iou_out: {:.4f}\n'.format(iou_out[-1]))
        fw.write('*' * 20 + '\n')


    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_in:  ' + str(np.mean(Dice_in)) + '\n')
    # fw.write('Dice_out: ' + str(np.mean(Dice_out)) + '\n')

    fw.write('Mean_HD\n')
    fw.write('hd_in:    ' + str(np.mean(hd_in)) + '\n')
    # fw.write('hd_out:   ' + str(np.mean(hd_out)) + '\n')

    fw.write('Mean_iou\n')
    fw.write('iou_in:   ' + str(np.mean(iou_in)) + '\n')
    # fw.write('iou_out:  ' + str(np.mean(iou_out)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_in))
    # dsc.append(np.mean(Dice_out))
    
    avg_hd = []
    avg_hd.append(np.mean(hd_in))
    # avg_hd.append(np.mean(hd_out))

    avg_iou = []
    avg_iou.append(np.mean(iou_in))
    # avg_iou.append(np.mean(iou_out))

    fw.write('DSC:  ' + str(np.mean(dsc)) + '\n')
    fw.write('HD:   ' + str(np.mean(avg_hd)) + '\n')
    fw.write('iou:  ' + str(np.mean(avg_iou)) + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        num_lines = len(lines)  
        # for line in lines:
        #     print(line)
        for line in lines[num_lines-13:]:
            print(line)


def test_two(args):
    mask_list = sorted(glob.glob(os.path.join(args.save_dir, 'mask', '*png')))
    pre_mask_list = sorted(glob.glob(os.path.join(args.save_dir, 'pre_mask', '*png')))

    Dice_in = []
    Dice_out = []

    hd_in = []
    hd_out = []

    iou_in = []
    iou_out = []

    def process_label(label):
        # mask_in = label == 160
        mask_out = label == 255
        return mask_out

    fw = open(args.save_dir + '/dice_pre.txt', 'a')
    for mask_path, pre_mask_path in zip(mask_list, pre_mask_list):
        # print(mask_path.split('/')[-1])
        # print(pre_mask_path.split('/')[-1])
        mask = np.array(Image.open(mask_path))
        pre_mask = np.array(Image.open(pre_mask_path))

        mask_in = process_label(mask)
        pre_mask_in = process_label(pre_mask)

        Dice_in.append(dice(mask_in, pre_mask_in))
        # Dice_out.append(dice(mask_out, pre_mask_out))

        hd_in.append(hd(mask_in, pre_mask_in))
        # hd_out.append(hd(mask_out, pre_mask_out))

        iou_in.append(iou(mask_in, pre_mask_in))
        # iou_out.append(iou(mask_out, pre_mask_out))

        fw.write('*' * 20 + '\n', )
        fw.write(pre_mask_path.split('/')[-1] + '\n')
        fw.write('Dice_in: {:.4f}\n'.format(Dice_in[-1]))
        # fw.write('Dice_out: {:.4f}\n'.format(Dice_out[-1]))
        fw.write('hd_in: {:.4f}\n'.format(hd_in[-1]))
        # fw.write('hd_out: {:.4f}\n'.format(hd_out[-1]))
        fw.write('iou_in: {:.4f}\n'.format(iou_in[-1]))
        # fw.write('iou_out: {:.4f}\n'.format(iou_out[-1]))
        fw.write('*' * 20 + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_in:  ' + str(np.mean(Dice_in)) + '\n')
    # fw.write('Dice_out: ' + str(np.mean(Dice_out)) + '\n')

    fw.write('Mean_HD\n')
    fw.write('hd_in:    ' + str(np.mean(hd_in)) + '\n')
    # fw.write('hd_out:   ' + str(np.mean(hd_out)) + '\n')

    fw.write('Mean_iou\n')
    fw.write('iou_in:   ' + str(np.mean(iou_in)) + '\n')
    # fw.write('iou_out:  ' + str(np.mean(iou_out)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_in))
    # dsc.append(np.mean(Dice_out))

    avg_hd = []
    avg_hd.append(np.mean(hd_in))
    # avg_hd.append(np.mean(hd_out))

    avg_iou = []
    avg_iou.append(np.mean(iou_in))
    # avg_iou.append(np.mean(iou_out))

    fw.write('DSC:  ' + str(np.mean(dsc)) + '\n')
    fw.write('HD:   ' + str(np.mean(avg_hd)) + '\n')
    fw.write('iou:  ' + str(np.mean(avg_iou)) + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        num_lines = len(lines)
        # for line in lines:
        #     print(line)
        for line in lines[num_lines - 13:]:
            print(line)
def test_IVUS109(args):
    mask_list = sorted(glob.glob(os.path.join(args.save_dir, 'mask', '*png')))
    pre_mask_list = sorted(glob.glob(os.path.join(args.save_dir, 'pre_mask', '*png')))

    Dice_in = []
    # Dice_out = []

    hd_in = []
    # hd_out = []

    iou_in = []
    # iou_out = []

    def process_label(label):
        mask_in = label == 255
        # mask_out = label == 80
        return mask_in#, mask_out

    fw = open(args.save_dir + '/dice_pre.txt', 'a')
    for mask_path, pre_mask_path in zip(mask_list, pre_mask_list):
        # print(mask_path.split('/')[-1])
        # print(pre_mask_path.split('/')[-1])
        mask = np.array(Image.open(mask_path))
        pre_mask = np.array(Image.open(pre_mask_path))

        mask_in = process_label(mask)
        pre_mask_in = process_label(pre_mask)

        Dice_in.append(dice(mask_in, pre_mask_in))
        # Dice_out.append(dice(mask_out, pre_mask_out))

        hd_in.append(hd(mask_in, pre_mask_in))
        # hd_out.append(hd(mask_out, pre_mask_out))

        iou_in.append(iou(mask_in, pre_mask_in))
        # iou_out.append(iou(mask_out, pre_mask_out))

        fw.write('*' * 20 + '\n', )
        fw.write(pre_mask_path.split('/')[-1] + '\n')
        fw.write('Dice_in: {:.4f}\n'.format(Dice_in[-1]))
        # fw.write('Dice_out: {:.4f}\n'.format(Dice_out[-1]))
        fw.write('hd_in: {:.4f}\n'.format(hd_in[-1]))
        # fw.write('hd_out: {:.4f}\n'.format(hd_out[-1]))
        fw.write('iou_in: {:.4f}\n'.format(iou_in[-1]))
        # fw.write('iou_out: {:.4f}\n'.format(iou_out[-1]))
        fw.write('*' * 20 + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_in:  ' + str(np.mean(Dice_in)) + '\n')
    # fw.write('Dice_out: ' + str(np.mean(Dice_out)) + '\n')
    #
    fw.write('Mean_HD\n')
    fw.write('hd_in:    ' + str(np.mean(hd_in)) + '\n')
    # fw.write('hd_out:   ' + str(np.mean(hd_out)) + '\n')

    fw.write('Mean_iou\n')
    fw.write('iou_in:   ' + str(np.mean(iou_in)) + '\n')
    # fw.write('iou_out:  ' + str(np.mean(iou_out)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_in))
    # dsc.append(np.mean(Dice_out))

    avg_hd = []
    avg_hd.append(np.mean(hd_in))
    # avg_hd.append(np.mean(hd_out))

    avg_iou = []
    avg_iou.append(np.mean(iou_in))
    # avg_iou.append(np.mean(iou_out))

    fw.write('DSC:  ' + str(np.mean(dsc)) + '\n')
    fw.write('HD:   ' + str(np.mean(avg_hd)) + '\n')
    fw.write('iou:  ' + str(np.mean(avg_iou)) + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        num_lines = len(lines)
        # for line in lines:
        #     print(line)
        for line in lines[num_lines - 13:]:
            print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    # args.save_dir = 'output_experiment/sam_unet_seg_ACDC_f0_tr_75'
    test_IVUS109(args)
