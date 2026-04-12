import os
import sys
import subprocess
import time

def run_script(script_path):
    """运行指定的 Python 脚本并捕获输出"""
    print(f"\n{'='*70}")
    print(f"🚀 Running: {os.path.basename(script_path)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # 获取当前 Python 解释器路径，确保在同一环境下运行
    python_exe = sys.executable
    
    # 启动子进程运行脚本
    try:
        # 使用 subprocess.run 运行，捕获 stdout 和 stderr
        # check=True 表示如果脚本返回非零退出码，将抛出异常
        result = subprocess.run([python_exe, script_path], check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Finished: {os.path.basename(script_path)} in {elapsed_time:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {os.path.basename(script_path)} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {os.path.basename(script_path)}: {e}")
        return False

def main():
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 按顺序定义的脚本列表 (01-04 是生成脚本)
    generation_pipeline = [
        "01_Generate_V_Base.py",
        "02_Generate_V_Geo.py",
        "03_Generate_V_Phys.py",
        "04_Generate_STS2R_Sim.py"
    ]
    
    # 后处理与特征生成脚本 (可选，但建议按顺序运行)
    post_pipeline = [
        "05_ROI_Filter_and_NPY_Converter.py",
        "06_Generate_Stage1.5_Offline_Data.py"
    ]
    
    total_start_time = time.time()
    success_scripts = []
    failed_scripts = []
    
    print("🔥 Starting STS2R Data Generation Pipeline (01-04) 🔥")
    
    # 运行主要的生成脚本 01-04
    for script_name in generation_pipeline:
        script_path = os.path.join(current_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"⚠️ Warning: Script {script_name} not found in {current_dir}. Skipping.")
            failed_scripts.append(script_name)
            continue
            
        if run_script(script_path):
            success_scripts.append(script_name)
        else:
            failed_scripts.append(script_name)
            # 如果中间一个脚本失败，询问是否继续？
            # 这里默认直接停止，以防产生错误的数据依赖
            print(f"\n🛑 Pipeline stopped due to failure in {script_name}.")
            break
    
    # 如果 01-04 成功，询问是否继续运行 05-06？
    # 考虑到自动化运行，这里默认继续运行 (或者你也可以选择只运行 01-04)
    if not failed_scripts and all(os.path.exists(os.path.join(current_dir, s)) for s in post_pipeline):
        print("\n" + "#"*70)
        print("🎉 Generation scripts (01-04) completed successfully!")
        print("🔄 Proceeding to Post-processing (05-06)...")
        print("#"*70)
        
        for script_name in post_pipeline:
            script_path = os.path.join(current_dir, script_name)
            if run_script(script_path):
                success_scripts.append(script_name)
            else:
                failed_scripts.append(script_name)
                break

    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("🏁 Pipeline Summary")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Success: {len(success_scripts)}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print(f"❌ Failed scripts: {', '.join(failed_scripts)}")
    else:
        print("✅ All scripts in the pipeline completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
