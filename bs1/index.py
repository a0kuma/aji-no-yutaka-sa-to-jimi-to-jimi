import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
dim=5
# ==========================================
# 1. 定義 4 層為單位的 Segment (對應 4+4 的 "4")
# ==========================================
class Segment4Layers(nn.Module):
        def __init__(self, dim=128):
                    super().__init__()
        # 每個 Segment 內部精確包含 4 層 Dummy Linear
                    self.layers = nn.Sequential(*[
                        nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) 
                                    for _ in range(4)
                                            ])

        def forward(self, x):
                # 💡 【核心實作】：直接在此處套用 Activation Checkpointing
        # 每 4 層作為一個 Checkpoint Block 重算單位
        # PyTorch 2.x 強烈建議使用 use_reentrant=False 以相容編譯與 Export
            def run_forward(tensor):
                        return self.layers(tensor)
        
            return checkpoint(run_forward, x, use_reentrant=False)

# ==========================================
# 2. 定義總共 32 層的完整模型
# ==========================================
class DummyModel32(nn.Module):
        def __init__(self, dim=128):
                    super().__init__()
        # 總共 32 層，被分成 8 個 Segment (8 * 4 = 32)
        # 稍後 Pipeline 切分時，每個 GPU 會分到 2 個 Segment (2 * 4 = 8 層)
                    self.segments = nn.ModuleList([
                        Segment4Layers(dim) for _ in range(8)
                                ])

        def forward(self, x):
                for segment in self.segments:
                                x = segment(x)
                return x

# ==========================================
# 3. 主訓練與 Pipeline 切分流程
# ==========================================
def main():
        # 啟動分散式環境 (4 個 GPU)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 參數設定
    dim = 128
    batch_size = 16
    n_microbatches = 4
    
    # 建立完整模型並搬到對應的 GPU
    model = DummyModel32(dim).to(device)
    
    # 為了讓 Pipeline API 追蹤 (Trace) 計算圖，需要準備一組 Micro-batch 大小的假資料
    mb_x = torch.randn(batch_size // n_microbatches, dim, device=device)
    
    # 💡 【Pipeline 切分點】：
    # 總共 8 個 segments (索引 0~7)
    # GPU 0: 負責 segments.0, segments.1  (共 8 層，內部自動拆為 4+4 兩個 checkpoint)
    # GPU 1: 負責 segments.2, segments.3  (共 8 層，內部自動拆為 4+4 兩個 checkpoint)
    # GPU 2: 負責 segments.4, segments.5  (共 8 層，內部自動拆為 4+4 兩個 checkpoint)
    # GPU 3: 負責 segments.6, segments.7  (共 8 層，內部自動拆為 4+4 兩個 checkpoint)
    split_spec = {
                    "segments.2": SplitPoint.BEGINNING,
                            "segments.4": SplitPoint.BEGINNING,
                                    "segments.6": SplitPoint.BEGINNING,
                                        }

    # 呼叫 pipeline 進行全圖追蹤與自動切分
    pipe = pipeline(
                    module=model,
                            mb_args=(mb_x,),
                                    split_spec=split_spec,
                                        )

    # 為當前的 Rank (GPU) 建立對應的 Pipeline Stage Runtime
    stage_index = rank
    stage = pipe.build_stage(stage_index, device)
    
    # 將 Stage 綁定到 GPipe 排程器
    schedule = ScheduleGPipe(stage, n_microbatches)

    # ==========================================
    # 4. 執行 Pipeline 前向傳播
    # ==========================================
    if rank == 0:
                # 只有 Stage 0 (GPU 0) 需要吃入完整的 Batch
        x = torch.randn(batch_size, dim, device=device, requires_grad=True)
        schedule.step(x)
    else:
                # 其他 Stage 會自動透過 P2P 網路接收前一個 GPU 的資料
        output = schedule.step()

    if rank == 3:
                # Stage 3 (GPU 3) 會產出最終結果
        print(f"[GPU {rank}] Pipeline 執行成功，最終輸出維度: {output.shape}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
        main()
