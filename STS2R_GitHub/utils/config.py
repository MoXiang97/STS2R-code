import torch

class Config:
    # 训练数据路径
    train_real_path = r"D:\Data\PaperAI\SIM2REAL\Shoe_Experiment_data\Train_Real_sim2real"
    train_sim_path = r"D:\Data\PaperAI\SIM2REAL\Shoe_Experiment_data\Train_shape"
    
    test_real_path = r"D:\Data\PaperAI\SIM2REAL\Shoe_Experiment_data\Train_Real_sim2real_test"
    
    # 模型参数
    num_points = 8192
    candidate_points = 1024
    batch_size = 8
    stage1_epochs = 50
    stage2_epochs = 30
    learning_rate = 0.001
    
    # 特征维度
    stage1_feature_dim = 10
    stage2_feature_dim = 10
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出目录
    output_dir = "./two_stage_output"
    
    # 随机种子
    seed = 42
