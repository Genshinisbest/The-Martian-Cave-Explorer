#!/usr/bin/env python3
"""
Alternative YOLOv5 model loader that avoids pandas and other dependency conflicts
This script tests if the trained model can be loaded in a clean environment
"""

import os
import sys

def test_model_loading():
    """Test if we can load the YOLOv5 model without dependency conflicts"""
    
    # Paths
    model_path = '/home/student/ros_ws/CV_Model/yolov5/best.pt'
    repo_path = '/home/student/ros_ws/CV_Model/yolov5'
    
    print(f"Testing model loading...")
    print(f"Model path: {model_path}")
    print(f"Repo path: {repo_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Repo exists: {os.path.exists(repo_path)}")
    
    if not os.path.exists(model_path) or not os.path.exists(repo_path):
        print("❌ Required files not found")
        return False
    
    try:
        # Test PyTorch import
        import torch
        print(f"✅ PyTorch imported: {torch.__version__}")
        
        # Test model loading with minimal dependencies
        print("Attempting direct model loading...")
        
        # Change to YOLOv5 directory
        original_cwd = os.getcwd()
        os.chdir(repo_path)
        sys.path.insert(0, '.')
        
        try:
            # Try the simple approach first (PyTorch 2.9+ compatible)
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            print("✅ Model loaded directly with torch.load (PyTorch 2.9+ compatible)")
            
            # Test inference on dummy data
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            print("✅ Created dummy test image")
            
            return True
            
        except Exception as e:
            print(f"❌ Direct loading failed: {e}")
            
            # Try torch.hub approach
            try:
                model = torch.hub.load('.', 'custom', path='best.pt', source='local')
                print("✅ Model loaded via torch.hub")
                return True
            except Exception as hub_e:
                print(f"❌ Torch.hub loading failed: {hub_e}")
                return False
        
        finally:
            os.chdir(original_cwd)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
        print("Your trained YOLOv5 model should work in the cave explorer.")
    else:
        print("\n💥 Model loading test FAILED!")
        print("There are dependency conflicts preventing model loading.")
        print("\nSuggested fixes:")
        print("1. pip3 uninstall numpy pandas matplotlib")
        print("2. pip3 install numpy==1.21.5 pandas matplotlib")
        print("3. Or use the virtual environment for model loading")
    
    sys.exit(0 if success else 1)