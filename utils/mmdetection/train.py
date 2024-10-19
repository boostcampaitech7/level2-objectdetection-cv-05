import argparse
import torch
import os

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='work dir(save logs and models) file path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume', help='path to latest checkpoint (if any)')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    
    # mmdetection 모듈 등록
    register_all_modules()
    
    # config 파일 등록
    cfg = Config.fromfile(args.config)
    
    # 인자로 받은 값 업데이트
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        
    # 장치 설정
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 체크포인트에서 학습 재개 설정
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # 설정 출력 (디버깅용)
    print(cfg.pretty_text)

    # Runner 생성 및 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()