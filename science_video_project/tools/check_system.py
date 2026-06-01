"""
系统检查工具 - 验证环境和依赖
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本... ", end="")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ 需要Python 3.8+，当前版本: {version.major}.{version.minor}")
        return False


def check_dependencies():
    """检查必需的依赖"""
    print("\n检查Python依赖:")
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        print(f"  - {name:30s} ", end="")
        try:
            __import__(module)
            print("✓")
        except ImportError:
            print("✗")
            all_ok = False
    
    return all_ok


def check_gpu():
    """检查GPU支持"""
    print("\n检查GPU支持:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  - CUDA可用: ✓")
            print(f"    设备: {torch.cuda.get_device_name(0)}")
            print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"  - CUDA不可用: 将使用CPU")
            return False
    except Exception as e:
        print(f"  - GPU检查失败: {e}")
        return False


def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构:")
    
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    required_files = {
        'app.py': '主应用程序',
        'requirements.txt': '依赖列表',
        'pipeline/config.py': '配置文件',
        'training/model_mvp.py': '模型定义',
        'frontend/index.html': 'Web界面',
    }
    
    all_ok = True
    for file_path, description in required_files.items():
        full_path = PROJECT_ROOT / file_path
        print(f"  - {description:20s} ({file_path:35s}) ", end="")
        if full_path.exists():
            print("✓")
        else:
            print("✗")
            all_ok = False
    
    return all_ok


def check_model_files():
    """检查模型文件"""
    print("\n检查模型文件:")
    
    try:
        from pipeline.config import CFG
        
        model_paths = {
            'checkpoint': CFG.checkpoint_dir / "best.pt",
            'text_model': Path(CFG.text_model_name),
            'clip_model': Path(CFG.clip_model_name),
            'asr_model': Path(CFG.asr_model_path),
        }
        
        all_ok = True
        for name, path in model_paths.items():
            print(f"  - {name:20s} ", end="")
            if path.exists():
                print(f"✓ ({path})")
            else:
                print(f"⚠ 不存在 ({path})")
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"  ✗ 无法加载配置: {e}")
        return False


def main():
    """主检查函数"""
    print("=" * 60)
    print("科学视频质量评分系统 - 系统检查工具")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("依赖安装", check_dependencies),
        ("GPU支持", check_gpu),
        ("项目结构", check_project_structure),
        ("模型文件", check_model_files),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"检查失败: {e}")
            results[name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结:")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name:20s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！系统已准备就绪")
        print("\n  启动服务: python app.py")
        print("  访问地址: http://localhost:5000")
    else:
        print("✗ 某些检查未通过，请修复问题后重试")
        print("\n  常见问题:")
        print("  1. 缺少依赖: pip install -r requirements.txt")
        print("  2. 模型文件缺失: 请检查配置中的路径")
        print("  3. GPU不可用: 将在config.py中device改为'cpu'")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
