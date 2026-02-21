# start.py
import os
import subprocess
import sys

def main():
    print("=" * 50)
    print("CAD Risk Prediction System")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('models/cad_model.pkl'):
        print("\n[1] Training ML model first time...")
        try:
            subprocess.run([sys.executable, "-c", 
                "from models.cad_ml_model import train_and_save_model; train_and_save_model()"])
            print("✓ Model trained successfully!")
        except Exception as e:
            print(f"✗ Error training model: {e}")
            print("  Will use simple model instead.")
    
    # Run Flask app
    print("\n[2] Starting Flask application...")
    print("   Access at: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
if __name__ == "__main__":
    main()