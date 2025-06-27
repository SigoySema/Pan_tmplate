#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:46:46
@Description: main.py
'''

from utils.config import get_config
from solver.unisolver import Solver
from solver.testsolver import Testsolver
# from solver.gf_solver import Solver
# from solver.midnsolver import Solver
# from solver.innformersolver import Solver
import argparse,os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'py-tra'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from metrics_runner import run_metrics

#训练测试一体化出结果_version
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    #parser.add_argument('--option_path', type=str, default='/root/Pan-Mamba/pan-sharpening/option.yml')
    parser.add_argument('--option_path', type=str, default='./option.yml')
    # parser.add_argument('--option_path', type=str, default='./option_GPU1.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
   # === 训练阶段 ===
    solver = Solver(cfg)
    log_name = solver.run()  # 修改 run() 返回 log_name

    # === 自动设置模型路径 ===    
    cfg['test']['algorithm'] = cfg['algorithm']
    cfg['test']['type'] = 'test'  # 或 'eval'
    cfg['test']['data_dir'] = cfg['data_dir_eval']
    cfg['test']['model'] = os.path.join(cfg['checkpoint'] + '/' + str(log_name), 'bestPSNR.pth')
    cfg['test']['save_dir'] = os.path.join(cfg['checkpoint'] + '/' + str(log_name),'result/')
    

    # === 测试阶段 ===
    test_solver = Testsolver(cfg)
    test_solver.run()


    # === 调用指标评估 ===
    path_ms = os.path.join(cfg['data_dir_eval'],'ms')          # 假设你在cfg里设置了这些
    path_pan = os.path.join(cfg['data_dir_eval'],'pan')    
    path_predict = os.path.join(cfg['test']['save_dir'],'test')  # 输出预测结果的路径

    run_metrics(path_ms, path_pan, path_predict,save_path =cfg['test']['save_dir'] )
    
