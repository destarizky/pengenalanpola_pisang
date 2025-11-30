"""
Test Script untuk Grad-CAM
Jalankan: python test_gradcam.py
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Configuration
MODEL_PATH = 'fish_classifier_model.keras'
IMG_SIZE = (150, 150)

def test_model_layers():
    """Test 1: Cek layer model"""
    print("="*60)
    print("TEST 1: Checking Model Layers")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file tidak ditemukan!")
        return False
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("\n📋 Available Layers:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name} ({layer.__class__.__name__})")
    
    # Check if 'last_conv' exists
    layer_names = [layer.name for layer in model.layers]
    if 'last_conv' in layer_names:
        print("\n✅ Layer 'last_conv' FOUND!")
        return True
    else:
        print("\n❌ Layer 'last_conv' NOT FOUND!")
        print("💡 Jalankan ulang notebook untuk rebuild model dengan layer 'last_conv'")
        return False

def test_gradcam_generation():
    """Test 2: Generate Grad-CAM"""
    print("\n" + "="*60)
    print("TEST 2: Generating Grad-CAM")
    print("="*60)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Create dummy image
    dummy_img = np.random.rand(150, 150, 3) * 255
    dummy_img = dummy_img.astype('uint8')
    img_pil = Image.fromarray(dummy_img)
    
    # Preprocess
    img_array = np.array(img_pil.resize(IMG_SIZE)).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"\n📊 Image shape: {img_array.shape}")
    
    try:
        # Create grad model
        last_conv_layer = model.get_layer('last_conv')
        # Use model.layers[-1].output instead of model.output to avoid list issue
        output_layer = model.layers[-1].output
        
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, output_layer]
        )
        
        print("✅ Grad model created successfully!")
        
        # Generate heatmap
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val != 0:
            heatmap = heatmap / max_val
        
        heatmap_np = heatmap.numpy()
        
        print(f"✅ Heatmap generated!")
        print(f"📊 Heatmap shape: {heatmap_np.shape}")
        print(f"📊 Heatmap range: [{heatmap_np.min():.3f}, {heatmap_np.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test 3: Test dengan gambar real dari test folder"""
    print("\n" + "="*60)
    print("TEST 3: Testing with Real Image")
    print("="*60)
    
    # Cari gambar test pertama
    test_dir = 'test'
    if not os.path.exists(test_dir):
        print("⚠️ Test directory tidak ditemukan")
        return False
    
    # Ambil gambar pertama yang ditemukan
    test_image_path = None
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image_path = os.path.join(root, file)
                break
        if test_image_path:
            break
    
    if not test_image_path:
        print("⚠️ Tidak ada gambar test ditemukan")
        return False
    
    print(f"\n📷 Using test image: {test_image_path}")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load and preprocess
        img = Image.open(test_image_path).convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_class_idx]
        
        print(f"✅ Prediction: Class {pred_class_idx} with confidence {confidence:.2%}")
        
        # Generate Grad-CAM
        last_conv_layer = model.get_layer('last_conv')
        # Use model.layers[-1].output instead of model.output to avoid list issue
        output_layer = model.layers[-1].output
        
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, output_layer]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val != 0:
            heatmap = heatmap / max_val
        
        heatmap_np = heatmap.numpy()
        
        # Create overlay
        heatmap_resized = cv2.resize(heatmap_np, IMG_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        print("✅ Grad-CAM berhasil dibuat untuk gambar real!")
        print(f"📊 Heatmap stats: min={heatmap_np.min():.3f}, max={heatmap_np.max():.3f}, mean={heatmap_np.mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔬 GRAD-CAM TEST SUITE")
    print("="*60)
    
    # Run tests
    test1_passed = test_model_layers()
    
    if test1_passed:
        test2_passed = test_gradcam_generation()
        test3_passed = test_with_real_image()
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Test 1 (Model Layers): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"Test 2 (Grad-CAM Gen): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        print(f"Test 3 (Real Image):   {'✅ PASSED' if test3_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed and test3_passed:
            print("\n🎉 ALL TESTS PASSED! Grad-CAM siap digunakan di Streamlit!")
        else:
            print("\n⚠️ Ada test yang gagal. Perbaiki issue di atas sebelum menjalankan Streamlit.")
    else:
        print("\n❌ Test 1 gagal. Perbaiki model terlebih dahulu.")
    
    print("="*60)
