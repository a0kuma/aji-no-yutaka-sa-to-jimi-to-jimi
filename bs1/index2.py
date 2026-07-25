import torch
import torch.nn as nn
from torchgpipe import GPipe

# ==========================================
# 1. 建立 32 層的模型
# ==========================================
# torchgpipe 嚴格要求模型必須是 nn.Sequential 的扁平結構
layers = []
for _ in range(32):
        layers.append(nn.Sequential(nn.Linear(128, 128), nn.ReLU()))

model = nn.Sequential(*layers)

# ==========================================
# 2. 定義切分策略 (Balance) 與 設備映射 (Devices)
# ==========================================
# balance: 定義每個 Partition (Segment) 包含幾層
# 我們要 4+4 的 checkpoint，所以把 32 層切成 8 個 Partition，每個 4 層
balance = [4, 4, 4, 4, 4, 4, 4, 4]

# devices: 定義這 8 個 Partition 分別要放在哪張 GPU 上
# 💡【核心解答】：這裡展現了在同一個 GPU 上放置多個 segment 的設定
devices = [
            'cuda:0', 'cuda:0',  # GPU 0 負責 Partition 0, 1 (共 8 層)
                'cuda:1', 'cuda:1',  # GPU 1 負責 Partition 2, 3 (共 8 層)
                    'cuda:2', 'cuda:2',  # GPU 2 負責 Partition 4, 5 (共 8 層)
                        'cuda:3', 'cuda:3',  # GPU 3 負責 Partition 6, 7 (共 8 層)
                        ]

# ==========================================
# 3. 使用 torchgpipe 包裝模型
# ==========================================
# chunks: 定義 Micro-batch 的數量 (例如 4)
# checkpoint='always': 開啟後，會自動在上述定義好的 8 個 Partition 獨立做 Checkpoint
model = GPipe(
            model, 
                balance=balance, 
                    devices=devices, 
                        chunks=4, 
                            checkpoint='always'
                            )

# ==========================================
# 4. 執行前向傳播
# ==========================================
def main():
        # 準備一個 Batch 的資料，並放在第一個 Partition 所在的設備上
    x = torch.randn(16, 128, requires_grad=True).to('cuda:0')
    
    # 直接呼叫 model 即可，內部會自動利用背景執行緒與 CUDA Stream 調度 4 張卡
    output = model(x)
    
    # 產出的 output 會自動落在最後一個 Partition 所在的設備 (cuda:3)
    print(f"Pipeline 執行成功，最終輸出維度: {output.shape}，位於 {output.device}")

if __name__ == "__main__":
        main()
