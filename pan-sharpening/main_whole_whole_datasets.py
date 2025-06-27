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

if __name__ == '__main__':
    
    # 外层循环，循环三次
    for i in range(3):
        print(f"\n GPU0 开始训练第 {i + 1} 轮...\n")
        parser = argparse.ArgumentParser(description='N_SR')
        parser.add_argument('--option_path', type=str, default='/root/Pan-Mamba/pan-sharpening')
        opt = parser.parse_args()
    
        # 配置文件列表，按顺序排列
        option_files = [
            'option_WV2.yml',
            'option_GF2.yml',
            'option_WV3.yml',
        ]
    
        # 遍历配置文件列表
        for option_file in option_files:
            # 获取每个配置文件的完整路径
            option_file_path = os.path.join(opt.option_path, option_file)
            print(f"Using config: {option_file_path}")
        
            # 加载配置
            cfg = get_config(option_file_path)

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




