# metrics_runner.py

import numpy as np
import os
import cv2
from PIL import Image
from metrics import ref_evaluate, no_ref_evaluate

def cal(ref, noref):
    reflist = []
    noreflist = []
    reflist.append(np.mean([ii[0] for ii in ref]))
    reflist.append(np.mean([ii[1] for ii in ref]))
    reflist.append(np.mean([ii[2] for ii in ref]))
    reflist.append(np.mean([ii[3] for ii in ref]))
    reflist.append(np.mean([ii[4] for ii in ref]))
    reflist.append(np.mean([ii[5] for ii in ref]))
    noreflist.append(np.mean([ih[0] for ih in noref]))
    noreflist.append(np.mean([ih[1] for ih in noref]))
    noreflist.append(np.mean([ih[2] for ih in noref]))
    return reflist, noreflist

def run_metrics(path_ms, path_pan, path_predict, save_path=''):
    save_path= save_path + 'metrics_result.txt'
    list_name = os.listdir(path_ms)
    list_ref = []
    list_noref = []
    fnb = 0

    for file_name_i in list_name:
        fnb += 1
        path_ms_file = os.path.join(path_ms, file_name_i)
        path_pan_file = os.path.join(path_pan, file_name_i)
        path_predict_file = os.path.join(path_predict, file_name_i)

        original_msi = np.array(Image.open(path_ms_file))
        original_pan = np.array(Image.open(path_pan_file))
        fused_image = np.array(Image.open(path_predict_file))

        gt = np.uint8(original_msi)
        used_ms = cv2.resize(original_msi, (original_msi.shape[1] // 4, original_msi.shape[0] // 4), cv2.INTER_CUBIC)
        used_pan = np.expand_dims(original_pan, -1)

        temp_ref_results = ref_evaluate(fused_image, gt)
        temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan), np.uint8(used_ms))
        list_ref.append(temp_ref_results)
        list_noref.append(temp_no_ref_results)

    temp_ref_results1, temp_no_ref_results1 = cal(list_ref, list_noref)

    ref_results = {'metrics: ': '  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q', 'deep': temp_ref_results1}
    no_ref_results = {'metrics: ': '  D_lamda,  D_s,    QNR', 'deep': temp_no_ref_results1}

    # 保存到 TXT
    with open(save_path, 'w') as f:
        f.write('################## reference comparision #######################\n')
        for index, i in enumerate(ref_results):
            if index == 0:
                f.write(f"{i}: {ref_results[i]}\n")
            else:
                f.write(f"{i}: {[round(j, 4) for j in ref_results[i]]}\n")

        f.write('################## no reference comparision ####################\n')
        for index, i in enumerate(no_ref_results):
            if index == 0:
                f.write(f"{i}: {no_ref_results[i]}\n")
            else:
                f.write(f"{i}: {[round(j, 4) for j in no_ref_results[i]]}\n")

    print(f"✔ 指标结果已保存至：{save_path}")
